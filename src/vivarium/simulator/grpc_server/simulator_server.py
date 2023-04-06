import grpc
import simulator_pb2_grpc
import simulator_pb2

from numproto import ndarray_to_proto, proto_to_ndarray
from protobuf_to_dict import protobuf_to_dict, dict_to_protobuf


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

    def GetSimulationConfig(self, request, context):
        config = simulator_pb2.SimulationConfig(box_size=self.simulator.simulation_config.box_size,
                                                map_dim=self.simulator.simulation_config.map_dim)
        return config

    def GetState(self, request, context):
        state = self.simulator.state
        positions = []
        thetas = []
        for ag in range(state.positions.shape[0]):
            positions += [simulator_pb2.Position(x=state.positions[ag, 0], y=state.positions[ag, 1])]
            thetas += [state.thetas[ag]]
        return simulator_pb2.State(positions=positions, thetas=thetas)

    def GetStateArray(self, request, context):
        positions = np.array(self.simulator.state.positions)
        return simulator_pb2.StateArray(array=ndarray_to_proto(positions))


    def Start(self, request, context):
        simulator.run(threaded=True)
        return Empty()

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
    simulation_config = config.SimulatorConfig(agent_config=agent_config)
    population_config = config.PopulationConfig()
    behavior_config = config.BehaviorConfig(population_config=population_config)

    simulator = Simulator(simulation_config=simulation_config, agent_config=agent_config,
                          behavior_config=behavior_config, population_config=population_config)
    logging.basicConfig()
    serve(simulator)
