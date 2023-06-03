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

from vivarium.simulator import config
from vivarium.simulator.simulator import Simulator, EngineConfig
from vivarium.simulator.sim_computation import dynamics_rigid

Empty = simulator_pb2.google_dot_protobuf_dot_empty__pb2.Empty


class SimulatorServerServicer(simulator_pb2_grpc.SimulatorServerServicer):
    def __init__(self, engine_config):
        self.engine_config = engine_config
        self.recorded_change_dict = defaultdict(dict)
        self._change_time = 0
        self._lock = Lock()

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
        serialized = config.param.serialize_parameters(subset=config.export_fields)
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

    def GetAgentConfigMessage(self, request, context):
        return simulator_pb2.AgentConfig(**self.simulator.agent_config.to_dict())

    def GetAgentConfigSerialized(self, request, context):
        config = self.engine_config.agent_configs[request.idx[0]]
        print('GetAgentConfigSerialized', config.to_dict())
        serialized = config.param.serialize_parameters(subset=config.export_fields)
        return simulator_pb2.AgentConfigSerialized(serialized=serialized)

    def GetAgentConfigsSerialized(self, request, context):
        self.engine_config.update_agent_configs_from_state()
        serialized = [config.param.serialize_parameters(subset=config.export_fields) for config in self.simulator.agent_configs]
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

    def GetNVEState(self, request, context):
        state = self.engine_config.simulator.state
        return simulator_pb2.NVEState(position=simulator_pb2.RigidBody(center=ndarray_to_proto(state.position.center),
                                                                       orientation=ndarray_to_proto(state.position.orientation)),
                                      momentum=simulator_pb2.RigidBody(center=ndarray_to_proto(state.momentum.center),
                                                                       orientation=ndarray_to_proto(state.momentum.orientation)),
                                      force=simulator_pb2.RigidBody(center=ndarray_to_proto(state.force.center),
                                                                    orientation=ndarray_to_proto(state.force.orientation)),
                                      mass=simulator_pb2.RigidBody(center=ndarray_to_proto(state.mass.center),
                                                                   orientation=ndarray_to_proto(state.mass.orientation)),
                                      idx=ndarray_to_proto(state.idx),
                                      prox=ndarray_to_proto(state.prox),
                                      motor=ndarray_to_proto(state.motor),
                                      behavior=ndarray_to_proto(state.behavior),
                                      wheel_diameter=ndarray_to_proto(state.wheel_diameter),
                                      base_length=ndarray_to_proto(state.base_length),
                                      speed_mul=ndarray_to_proto(state.speed_mul),
                                      theta_mul=ndarray_to_proto(state.theta_mul),
                                      proxs_dist_max=ndarray_to_proto(state.proxs_dist_max),
                                      proxs_cos_min=ndarray_to_proto(state.proxs_cos_min),
                                      color=ndarray_to_proto(state.color),
                                      entity_type=ndarray_to_proto(state.entity_type)
                                      )

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
            print('SetAgentConfig', d)

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
            idx = np.array(request.agent_idx.idx)
            self.engine_config.simulator.set_state(idx, request.nested_field, proto_to_ndarray(request.value))
        return Empty()

def serve(engine_config):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    simulator_pb2_grpc.add_SimulatorServerServicer_to_server(
        SimulatorServerServicer(engine_config), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    simulation_config = config.SimulatorConfig(to_jit=True, dynamics_fn=dynamics_rigid)

    engine_config = EngineConfig(simulation_config=simulation_config)

    print('Simulator server started')
    logging.basicConfig()
    serve(engine_config)
