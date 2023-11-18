import json

from collections import defaultdict

from numproto.numproto import ndarray_to_proto, proto_to_ndarray

import grpc
import simulator_pb2_grpc
import simulator_pb2

import numpy as np
import logging
from concurrent import futures
from threading import Lock
from contextlib import contextmanager

from vivarium.controllers import config
from vivarium.simulator.simulator import EngineConfig
from vivarium.simulator.sim_computation import dynamics_rigid
import vivarium.simulator.behaviors as behaviors
from vivarium.simulator.grpc_server import converters


# import dill

Empty = simulator_pb2.google_dot_protobuf_dot_empty__pb2.Empty


@contextmanager
def nonblocking(lock):
    locked = lock.acquire(False)
    try:
        yield locked
    finally:
        if locked:
            lock.release()


class SimulatorServerServicer(simulator_pb2_grpc.SimulatorServerServicer):
    def __init__(self, engine_config):
        self.engine_config = engine_config
        self.recorded_change_dict = defaultdict(dict)
        self._change_time = 0
        self._simulation_time = 0
        self._lock = Lock()
        self.engine_config.simulator.subscribe(self)
        self._stream_started = False

    def notify(self, simulation_time):
        self._simulation_time = simulation_time

    def _record_change(self, name, **kwargs):
        for k in self.recorded_change_dict.keys():
            if k != name:
                self.recorded_change_dict[k].update(kwargs)
        self._change_time += 1

    def GetChangeTime(self, request, context):
        return simulator_pb2.Time(time=self._change_time)

    def GetSimulationConfigMessage(self, request, context):
        config = simulator_pb2.SimulationConfig(**self.engine_config.simulation_config.to_dict())
        return config

    def GetSimulationConfigSerialized(self, request, context):
        config = self.engine_config.simulation_config
        serialized = config.param.serialize_parameters(subset=config.param_names())
        return simulator_pb2.SimulationConfigSerialized(serialized=serialized)

    def GetRecordedChanges(self, request, context):
        d = self.recorded_change_dict[request.name]
        has_entity_behaviors = 'entity_behaviors' in d
        if has_entity_behaviors:
            entity_behaviors = ndarray_to_proto(d['entity_behaviors'])
            del d['entity_behaviors']
        else:
            entity_behaviors = ndarray_to_proto(np.array(0))
        serialized_dict = json.dumps(d)
        self.recorded_change_dict[request.name] = {}
        return simulator_pb2.SerializedDict(serialized_dict=serialized_dict,
                                            has_entity_behaviors=has_entity_behaviors, entity_behaviors=entity_behaviors)

    def GetAllConfigSerialized(self, request, context):
        configs = dict(simulation_config=self.engine_config.simulation_config,
                       agent_configs=self.engine_config.agent_configs,
                       object_configs=self.engine_config.object_configs)

    def GetAgentConfigMessage(self, request, context):
        return simulator_pb2.AgentConfig(**self.simulator.agent_config.to_dict())

    def GetAgentConfigSerialized(self, request, context):
        config = self.engine_config.agent_configs[request.idx[0]]
        print('GetAgentConfigSerialized', config.to_dict())
        serialized = config.param.serialize_parameters(subset=config.param_names())
        return simulator_pb2.AgentConfigSerialized(serialized=serialized)

    def GetAgentConfigsSerialized(self, request, context):
        self.engine_config.update_agent_configs_from_state()
        serialized = [config.param.serialize_parameters(subset=config.param_names()) for config in self.simulator.agent_configs]
        # self._change_time += 1
        return simulator_pb2.AgentConfigsSerialized(serialized=serialized)

    def GetAgentConfigs(self, request, context):
        res = []
        for config in self.engine_config.agent_configs:
            d = config.to_dict()
            res.append(simulator_pb2.AgentConfig(**d))
        return simulator_pb2.AgentConfigs(agent_configs=res)

    def GetAgentConfig(self, request, context):
        config = self.engine_config.agent_configs[request.idx]

    def GetStateMessage(self, request, context):
        state = self.engine_config.simulator.state
        x = state.position[:, 0]
        y = state.position[:, 1]
        thetas = state.thetas
        return simulator_pb2.State(positions=simulator_pb2.Position(x=x, y=y), thetas=thetas)

    def GetStateArrays(self, request, context):
        state = self.engine_config.simulator.state
        return simulator_pb2.StateArrays(positions=ndarray_to_proto(state.position),
                                         thetas=ndarray_to_proto(state.theta),
                                         proxs=ndarray_to_proto(state.prox),
                                         motors=ndarray_to_proto(state.motor),
                                         entity_type=state.entity_type)

    def GetState(self, request, context):
        state = self.engine_config.simulator.state
        return converters.state_to_proto(state)

    def GetNVEState(self, request, context):
        nve_state = self.engine_config.simulator.state.nve_state
        return converters.nve_state_to_proto(nve_state)

    def GetAgentState(self, request, context):
        agent_state = self.engine_config.simulator.state.agent_state
        return converters.agent_state_to_proto(agent_state)

    def GetObjectState(self, request, context):
        object_state = self.engine_config.simulator.state.object_state
        return converters.object_state_to_proto(object_state)

    def Start(self, request, context):
        self.engine_config.simulator.run(threaded=True)
        return Empty()

    def IsStarted(self, request, context):
        return simulator_pb2.IsStartedState(is_started=self.engine_config.simulator.is_started)

    def Stop(self, request, context):
        self.engine_config.simulator.stop()
        return Empty()

    def SetSimulationConfig(self, request, context):
        with self._lock:
            d = json.loads(request.dict.serialized)
            print('SetSimulationConfig', d)
            with self.engine_config.simulator.pause():
                self.engine_config.simulation_config.param.update(**d)
            self._record_change(request.name.name, **d)
        return Empty()

    def SetAgentConfig(self, request, context):
        with self._lock:
            d = json.loads(request.dict.serialized)
            print('SetAgentConfig', request.idx, d)

            with self.engine_config.simulator.pause():
                for idx in request.idx.idx:
                    self.engine_config.agent_configs[idx].param.update(**d)

            self._record_change(request.name.name, **d)
        return Empty()

    def SetSimulationConfigSerialized(self, request, context):
        serialized = request.serialized
        conf = config.SimulatorConfig(**config.SimulatorConfig.param.deserialize_parameters(serialized))
        self.engine_config.simulation_config.param.update(**conf.to_dict())
        return Empty()

    def SetMotors(self, request, context):
        with self._lock:
            for idx in request.agent_idx.idx:
                self.engine_config.simulator.set_motors(idx, request.motor_idx, request.value)
        return Empty()

    def SetState(self, request, context):

        with self._lock:
            nve_idx = np.array(request.nve_idx)
            col_idx = np.array(request.col_idx)
            # with self.engine_config.simulator.pause():
            self.engine_config.simulator.set_state(request.nested_field, nve_idx, col_idx,
                                                   proto_to_ndarray(request.value))
        return Empty()

    def SensoryMotorStream(self, request_iterator, context):
        assert not self.engine_config.simulator.is_started
        for motor in request_iterator:
            print('motor', motor.motor)
            self.engine_config.simulator.set_state(('motor',), np.array([motor.agent_idx]), np.arange(2),
                                                   np.array([motor.motor]))
            self.engine_config.simulator.run(threaded=False, num_loops=1)
            prox = self.engine_config.simulator.state.prox[motor.agent_idx].tolist()
            print('prox', prox)
            yield simulator_pb2.Prox(agent_idx=motor.agent_idx,
                                     prox=prox)

    def NVEStateStream(self, request, context):
        assert not self.engine_config.simulator.is_started
        self._stream_started = True
        t = 0
        while self._stream_started:
            print('t = ', t)
            self.engine_config.simulator.run(threaded=False, num_loops=1)
            t += 1
            yield self.GetNVEState(request, context)

    def StopNVEStream(self, request, context):
        self._stream_started = False

    def AgentStep(self, request, context):
        assert not self.engine_config.simulator.is_started
        self.engine_config.simulator.set_state(('motor',), np.array([request.agent_idx]), np.arange(2),
                                               np.array([request.motor]))
        self.engine_config.simulator.run(threaded=False, num_loops=1)
        prox = self.engine_config.simulator.state.prox[request.agent_idx].tolist()
        return simulator_pb2.Prox(agent_idx=request.agent_idx,
                                  prox=prox)

    def Step(self, request, context):
        assert not self.engine_config.simulator.is_started
        self.engine_config.simulator.run(threaded=False, num_loops=1)
        return self.GetState(None, None)

    def StartBehavior(self, request, context):
        assert not self.engine_config.simulator.is_started
        behavior_function = dill.loads(request.function)
        for t in range(100):
            prox = self.engine_config.simulator.state.prox[request.agent_idx].tolist()
            motor = behavior_function(prox)
            self.engine_config.simulator.set_state(('motor',), np.array([request.agent_idx]), np.arange(2),
                                                   np.array([motor]))
            self.engine_config.simulator.run(threaded=False, num_loops=1)

        return Empty()

    def AddAgents(self, request, context):
        with self._lock:
            d = json.loads(request.serialized_config)

            prev_num_agents = len(self.engine_config.agent_configs)
            new_configs = []
            for i in range(request.n_agents):
                d['x_position'] += np.random.randn() * self.engine_config.simulation_config.box_size / 100.
                d['y_position'] += np.random.randn() * self.engine_config.simulation_config.box_size / 100.
                d['idx'] = prev_num_agents + i
                new_configs.append(config.AgentConfig(**d))

            for c in new_configs:
                c.param.watch(self.engine_config.update_state, list(c.param_names()), onlychanged=True)

            with self.engine_config.simulator.pause():
                self.engine_config.agent_configs += new_configs
                self.engine_config.simulation_config.n_agents += request.n_agents
            self._change_time += 1
        return simulator_pb2.AgentIdx(idx=[c.idx for c in self.engine_config.agent_configs])

    def RemoveAgents(self, request, context):
        with self._lock:
            with self.engine_config.simulator.pause():
                for i in request.idx:
                    self.engine_config.agent_configs.pop(i)
                for i, c in enumerate(self.engine_config.agent_configs):
                    c.idx = i
                self.engine_config.simulation_config.n_agents -= len(request.idx)
        self._change_time += 1
        return Empty()


def serve(engine_config):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    simulator_pb2_grpc.add_SimulatorServerServicer_to_server(
        SimulatorServerServicer(engine_config), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    simulation_config = config.SimulatorConfig(to_jit=True)

    engine_config = EngineConfig(dynamics_fn=dynamics_rigid, behavior_bank=behaviors.behavior_bank,
                                 simulation_config=simulation_config)

    #engine_config.param.agent_configs.objects = [engine_config.param.agent_config.objects[0]]

    # new_configs = engine_config.param.agent_configs.objects + [config.AgentConfig(idx=100)
    #                                          for i in range(1)]
    # engine_config.param.agent_configs.objects = new_configs



    # for idx in range(1, 10):
    #     engine_config.agent_configs[idx].behavior = 'noop'

    # engine_config.simulator.set_state(('behavior', ), np.arange(2), None, 1 * np.ones(2, dtype=int))
    # engine_config.simulator.set_state(('position', 'center'), np.arange(2), np.arange(2), np.array([[30., 50.], [70., 50.]]))
    # engine_config.simulator.set_state(('position', 'orientation'), np.arange(2), None, np.array([np.pi / 8., 9 * np.pi / 8.]))
    #
    # engine_config.simulator.run(threaded=False, num_loops=10)

    print('Simulator server started')
    logging.basicConfig()
    serve(engine_config)



