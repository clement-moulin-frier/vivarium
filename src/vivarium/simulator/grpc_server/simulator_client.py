import grpc
from vivarium.simulator.grpc_server import simulator_pb2_grpc
import vivarium.simulator.grpc_server.simulator_pb2 as simulator_pb2

from protobuf_to_dict import protobuf_to_dict, dict_to_protobuf

Empty = simulator_pb2.google_dot_protobuf_dot_empty__pb2.Empty

# def message_to_dict(message):
#

class SimulatorGRPCClient:
    def __init__(self):
        channel = grpc.insecure_channel('localhost:50051')
        self.stub = simulator_pb2_grpc.SimulatorServerStub(channel)

    def get_config(self):
        config = self.stub.GetSimulationConfig(Empty())
        return protobuf_to_dict(config)

    def get_agent_config(self):
        config = self.stub.GetAgentConfig(Empty())
        return protobuf_to_dict(config)

    def get_population_config(self):
        config = self.stub.GetPopulationConfig(Empty())
        return protobuf_to_dict(config)

    def set_population_config(self, **kwargs):
        config = simulator_pb2.PopulationConfig(**kwargs)
        self.stub.SetPopulationConfig(config)
