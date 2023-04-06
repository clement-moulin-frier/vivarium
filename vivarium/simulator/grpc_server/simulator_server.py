from numproto.numproto import ndarray_to_proto, proto_to_ndarray
from protobuf_to_dict import protobuf_to_dict, dict_to_protobuf

import grpc
import simulator_pb2_grpc
import simulator_pb2


# from numproto import ndarray_to_proto, proto_to_ndarray
#
import numpy as np
import logging
from concurrent import futures

from vivarium.simulator import config
from vivarium.simulator.simulator import Simulator

Empty = simulator_pb2.google_dot_protobuf_dot_empty__pb2.Empty

class SimulatorServerServicer(simulator_pb2_grpc.SimulatorServerServicer):
    def __init__(self, simulator):
        self.simulator = simulator

    def GetSimulationConfigMessage(self, request, context):
        config = simulator_pb2.SimulationConfig(**self.simulator.simulation_config.to_dict())
        return config

    def GetSimulationConfigSerialized(self, request, context):
        config = self.simulator.simulation_config
        serialized = config.param.serialize_parameters(subset=config.export_fields)
        return simulator_pb2.SimulationConfigSerialized(serialized=serialized)

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
        x = state.positions[:, 0]
        y = state.positions[:, 1]
        thetas = state.thetas
        return simulator_pb2.State(positions=simulator_pb2.Position(x=x, y=y), thetas=thetas)

    def GetStateArrays(self, request, context):
        state = self.simulator.state
        return simulator_pb2.StateArrays(positions=ndarray_to_proto(state.positions),
                                         thetas=ndarray_to_proto(state.thetas),
                                         proxs=ndarray_to_proto(state.proxs),
                                         motors=ndarray_to_proto(state.motors),
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
    population_config = config.PopulationConfig()
    # behavior_config = config.BehaviorConfig(population_config=population_config)

    simulator = Simulator(simulation_config=simulation_config, agent_config=agent_config,
                          population_config=population_config)
    logging.basicConfig()
    serve(simulator)
