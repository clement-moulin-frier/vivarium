import grpc
from vivarium.simulator.grpc_server import simulator_pb2_grpc
import vivarium.simulator.grpc_server.simulator_pb2 as simulator_pb2

from vivarium.simulator.config import SimulatorConfig, AgentConfig, PopulationConfig
from vivarium.simulator.sim_computation import Population

from numproto.numproto import ndarray_to_proto, proto_to_ndarray

from protobuf_to_dict import protobuf_to_dict, dict_to_protobuf

Empty = simulator_pb2.google_dot_protobuf_dot_empty__pb2.Empty

# def message_to_dict(message):
#

class SimulatorGRPCClient:
    def __init__(self):
        channel = grpc.insecure_channel('localhost:50051')
        self.stub = simulator_pb2_grpc.SimulatorServerStub(channel)

    def start(self):
        self.stub.Start(Empty())

    def stop(self):
        self.stub.Stop(Empty())

    def get_sim_config_dict(self):
        config = self.stub.GetSimulationConfig(Empty())
        return protobuf_to_dict(config)

    def get_sim_config(self):
        serialized = self.stub.GetSimulationConfigSerialized(Empty()).serialized
        return SimulatorConfig(**SimulatorConfig.param.deserialize_parameters(serialized))

    def get_agent_config_dict(self):
        config = self.stub.GetAgentConfig(Empty())
        return protobuf_to_dict(config)

    def get_agent_config(self):
        serialized = self.stub.GetAgentConfigSerialized(Empty()).serialized
        return AgentConfig(**AgentConfig.param.deserialize_parameters(serialized))

    def get_population_config_dict(self):
        config = self.stub.GetPopulationConfig(Empty())
        return protobuf_to_dict(config)

    def get_population_config(self):
        serialized = self.stub.GetPopulationConfigSerialized(Empty()).serialized
        return PopulationConfig(**PopulationConfig.param.deserialize_parameters(serialized))

    def set_population_config(self, population_config):
        config = simulator_pb2.PopulationConfig(**population_config.to_dict())
        self.stub.SetPopulationConfig(config)

    def get_state(self):
        return self.stub.GetStateMessage(Empty())

    def get_state_arrays(self):
        state = self.stub.GetStateArrays(Empty())
        return Population(positions=proto_to_ndarray(state.positions),
                          thetas=proto_to_ndarray(state.thetas),
                          proxs=proto_to_ndarray(state.proxs),
                          motors=proto_to_ndarray(state.motors),
                          entity_type=state.entity_type)

    def is_started(self):
        return self.stub.IsStarted(Empty()).is_started

