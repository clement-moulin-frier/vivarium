import grpc
from numproto.numproto import ndarray_to_proto
import vivarium.simulator.grpc_server.simulator_pb2 as simulator_pb2

from vivarium.simulator.grpc_server import simulator_pb2_grpc
from vivarium.simulator.grpc_server.simulator_client_abc import SimulatorClient
from vivarium.simulator.grpc_server.converters import proto_to_state, proto_to_nve_state, proto_to_agent_state, proto_to_object_state

Empty = simulator_pb2.google_dot_protobuf_dot_empty__pb2.Empty


class SimulatorGRPCClient(SimulatorClient):
    """A client for the simulator server that uses gRPC.

    :param SimulatorClient: Abstract base class for simulator clients.
    """
    def __init__(self, name=None):
        self.name = name
        channel = grpc.insecure_channel('localhost:50051')
        self.stub = simulator_pb2_grpc.SimulatorServerStub(channel)
        self.streaming_started = False
        self.state = self.get_state()
        self.scene_name = self.get_scene_name()
        self.subtypes_labels = self.get_subtypes_labels()

    def start(self):
        self.stub.Start(Empty())

    def stop(self):
        self.stub.Stop(Empty())

    def get_change_time(self):
        return self.stub.GetChangeTime(Empty()).time

    def set_state(self, nested_field, ent_idx, column_idx, value):
        state_change = simulator_pb2.StateChange(
            nested_field=nested_field, 
            ent_idx=ent_idx, col_idx=column_idx,
            value=ndarray_to_proto(value)
        )
        self.stub.SetState(state_change)

    def get_state(self):
        state = self.stub.GetState(Empty())
        return proto_to_state(state)

    def get_nve_state(self):
        entity_state = self.stub.GetNVEState(Empty())
        return proto_to_nve_state(entity_state)

    def get_agent_state(self):
        agent_state = self.stub.GetAgentState(Empty())
        return proto_to_agent_state(agent_state)

    def get_object_state(self):
        object_state = self.stub.GetObjectState(Empty())
        return proto_to_object_state(object_state)
    
    def get_scene_name(self):
        response = self.stub.GetSceneName(Empty())
        scene_name = response.scene_name
        return scene_name
    
    def get_subtypes_labels(self):
        response = self.stub.GetSubtypesLabels(Empty())
        subtype_labels_dict = dict(response.data)
        return subtype_labels_dict

    def step(self):
        self.state = proto_to_state(self.stub.Step(Empty()))
        return self.state

    def is_started(self):
        return self.stub.IsStarted(Empty()).is_started
