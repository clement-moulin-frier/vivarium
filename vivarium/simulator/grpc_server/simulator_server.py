import json
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
from vivarium.simulator.simulator import Simulator, VerletSimulator

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
        config = self.simulator.agent_config
        serialized = config.param.serialize_parameters(subset=config.export_fields)
        return simulator_pb2.AgentConfigSerialized(serialized=serialized)

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
        if 'entity_behaviors' in d_conf:
            entity_behaviors = proto_to_ndarray(request.config.entity_behaviors)
            d_conf['entity_behaviors'] = entity_behaviors
        print('SetSimulationConfig', d_conf)
        self.simulator.simulation_config.param.update(**d_conf)
        self._record_change(request.name.name, **d_conf)
        return Empty()

    def SetSimulationConfigSerialized(self, request, context):
        serialized = request.serialized
        # print('SetSimulationConfigSerialized', serialized)
        conf = config.SimulatorConfig(**config.SimulatorConfig.param.deserialize_parameters(serialized))
        self.simulator.simulation_config.param.update(**conf.to_dict())
        return Empty()

    def UpdateNeighborFn(self, request, context):
        self.simulator.update_neighbor_fn()
        return Empty()
    def UpdateStateNeighbors(self, request, context):
        self.simulator.update_state_neighbors()
        return Empty()
    def UpdateFunctionUpdate(self, request, context):
        self.simulator.update_function_update()
        return Empty()
    def UpdateBehaviors(self, request, context):
        self.simulator.update_behaviors()
        return Empty()


def serve(simulator):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    simulator_pb2_grpc.add_SimulatorServerServicer_to_server(
        SimulatorServerServicer(simulator), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    agent_config = config.AgentConfig()
    simulation_config = config.SimulatorConfig()
    # simulation_config.to_jit = False
    # population_config = config.PopulationConfig()
    engine_config = config.EngineConfig()
    # behavior_config = config.BehaviorConfig(population_config=population_config)

    simulator = VerletSimulator(simulation_config=simulation_config, agent_config=agent_config,
                                engine_config=engine_config)

    print('Done?')
    # simulator.init_simulator()
    logging.basicConfig()
    serve(simulator)
