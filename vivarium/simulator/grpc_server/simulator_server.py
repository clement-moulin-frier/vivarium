import json
import time
from collections import defaultdict

from numproto.numproto import ndarray_to_proto, proto_to_ndarray
from protobuf_to_dict import protobuf_to_dict, dict_to_protobuf

import grpc
import simulator_pb2_grpc
import simulator_pb2

import numpy as np
import logging
from concurrent import futures

from vivarium.simulator import config
from vivarium.simulator.simulator import Simulator, EngineConfig
from vivarium.simulator.sim_computation import dynamics_rigid

Empty = simulator_pb2.google_dot_protobuf_dot_empty__pb2.Empty


class SimulatorServerServicer(simulator_pb2_grpc.SimulatorServerServicer):
    def __init__(self, simulator):
        self.simulator = simulator
        self.recorded_change_dict = defaultdict(dict)
        self._change_time = 0

    def _record_change(self, name, **kwargs):
        for k in self.recorded_change_dict.keys():
            if k != name:
                self.recorded_change_dict[k].update(kwargs)
        self._change_time += 1

    def GetChangeTime(self, request, context):
        return simulator_pb2.Time(time=self._change_time)

    def GetSimulationConfigMessage(self, request, context):
        config = simulator_pb2.SimulationConfig(**self.simulator.simulation_config.to_dict())
        return config

    def GetSimulationConfigSerialized(self, request, context):
        config = self.simulator.simulation_config
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
        config = self.simulator.agent_configs[request.idx]
        serialized = config.param.serialize_parameters(subset=config.export_fields)
        return simulator_pb2.AgentConfigSerialized(serialized=serialized)

    def GetAgentConfigs(self, request, context):
        res = []
        for config in self.simulator.agent_configs:
            d = config.to_dict()
            res.append(simulator_pb2.AgentConfig(**d))
        return simulator_pb2.AgentConfigs(agent_configs=res)

    def GetAgentConfig(self, request, context):
        config = self.simulator.agent_configs[request.idx]

    def GetPopulationConfigMessage(self, request, context):
        return simulator_pb2.PopulationConfig(**self.simulator.population_config.to_dict())

    def GetPopulationConfigSerialized(self, request, context):
        config = self.simulator.population_config
        serialized = config.param.serialize_parameters(subset=config.export_fields)
        return simulator_pb2.PopulationConfigSerialized(serialized=serialized)

    def GetStateMessage(self, request, context):
        state = self.simulator.state
        x = state.position[:, 0]
        y = state.position[:, 1]
        thetas = state.thetas
        return simulator_pb2.State(positions=simulator_pb2.Position(x=x, y=y), thetas=thetas)

    def GetStateArrays(self, request, context):
        state = self.simulator.state
        return simulator_pb2.StateArrays(positions=ndarray_to_proto(state.position),
                                         thetas=ndarray_to_proto(state.theta),
                                         proxs=ndarray_to_proto(state.prox),
                                         motors=ndarray_to_proto(state.motor),
                                         entity_type=state.entity_type)

    def GetNVEState(self, request, context):
        state = self.simulator.state
        return simulator_pb2.NVEState(position=simulator_pb2.RigidBody(center=ndarray_to_proto(state.position.center),
                                                                       orientation=ndarray_to_proto(state.position.orientation)),
                                      momentum=simulator_pb2.RigidBody(center=ndarray_to_proto(state.momentum.center),
                                                                       orientation=ndarray_to_proto(state.momentum.orientation)),
                                      force=simulator_pb2.RigidBody(center=ndarray_to_proto(state.force.center),
                                                                    orientation=ndarray_to_proto(state.force.orientation)),
                                      mass=simulator_pb2.RigidBody(center=ndarray_to_proto(state.mass.center),
                                                                   orientation=ndarray_to_proto(state.mass.orientation)),
                                      prox=ndarray_to_proto(state.prox),
                                      motor=ndarray_to_proto(state.motor),
                                      behavior=ndarray_to_proto(state.behavior),
                                      wheel_diameter=ndarray_to_proto(state.wheel_diameter),
                                      base_length=ndarray_to_proto(state.base_length),
                                      speed_mul=ndarray_to_proto(state.speed_mul),
                                      theta_mul=ndarray_to_proto(state.theta_mul),
                                      proxs_dist_max=ndarray_to_proto(state.proxs_dist_max),
                                      proxs_cos_min=ndarray_to_proto(state.proxs_cos_min),
                                      entity_type=ndarray_to_proto(state.entity_type)
                                      )

    def Start(self, request, context):
        simulator.run(threaded=True)
        return Empty()

    def IsStarted(self, request, context):
        return simulator_pb2.IsStartedState(is_started=simulator.is_started)

    def Stop(self, request, context):
        self.simulator.stop()
        return Empty()

    def SetPopulationConfig(self, request, context):
        d_pop = protobuf_to_dict(request)
        self.simulator.population_config.param.update(**d_pop)
        return Empty()

    def SetSimulationConfig(self, request, context):
        d_conf = protobuf_to_dict(request.config)
        print('SetSimulationConfig', d_conf)
        self.simulator.simulation_config.param.update(**d_conf)
        self._record_change(request.name.name, **d_conf)
        return Empty()

    def SetAgentConfig(self, request, context):
        d_conf = protobuf_to_dict(request.config)
        print('SetAgentConfig', d_conf)

        with self.simulator.pause() as s:
            s.agent_configs[request.idx.idx].param.update(**d_conf)

        self._record_change(request.name.name, **d_conf)
        return Empty()

    def SetSimulationConfigSerialized(self, request, context):
        serialized = request.serialized
        conf = config.SimulatorConfig(**config.SimulatorConfig.param.deserialize_parameters(serialized))
        self.simulator.simulation_config.param.update(**conf.to_dict())
        return Empty()

    def SetMotors(self, request, context):
        ag = request.agent_slice
        self.simulator.set_motors(slice(ag.start, ag.stop, 1 if ag.step == 0 else ag.step),
                                  proto_to_ndarray(request.motors))
        return Empty()

def serve(simulator):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    simulator_pb2_grpc.add_SimulatorServerServicer_to_server(
        SimulatorServerServicer(simulator), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    simulation_config = config.SimulatorConfig()

    engine_config = EngineConfig(simulation_config=simulation_config, dynamics_fn=dynamics_rigid)

    simulator = Simulator(engine_config=engine_config)

    print('Simulator server started')
    logging.basicConfig()
    serve(simulator)
