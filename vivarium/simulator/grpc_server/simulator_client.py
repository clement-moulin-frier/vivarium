import json

import grpc
from vivarium.simulator.grpc_server import simulator_pb2_grpc
import vivarium.simulator.grpc_server.simulator_pb2 as simulator_pb2

from vivarium.controllers.config import SimulatorConfig, AgentConfig
from vivarium.simulator.grpc_server.converters import proto_to_state, proto_to_nve_state, proto_to_agent_state, \
    proto_to_object_state
from vivarium.simulator.grpc_server.simulator_client_abc import SimulatorClient

from numproto.numproto import ndarray_to_proto, proto_to_ndarray

# import dill
import threading

Empty = simulator_pb2.google_dot_protobuf_dot_empty__pb2.Empty


class SimulatorGRPCClient(SimulatorClient):
    def __init__(self, name=None):
        self.name = name
        channel = grpc.insecure_channel('localhost:50051')
        self.stub = simulator_pb2_grpc.SimulatorServerStub(channel)
        self.streaming_started = False
        self.state = self.get_state()

    def start(self):
        self.stub.Start(Empty())

    def stop(self):
        self.stub.Stop(Empty())

    def get_change_time(self):
        return self.stub.GetChangeTime(Empty()).time

    def set_state(self, nested_field, nve_idx, column_idx, value):
        state_change = simulator_pb2.StateChange(nested_field=nested_field, nve_idx=nve_idx, col_idx=column_idx,
                                                 value=ndarray_to_proto(value))
        self.stub.SetState(state_change)

    def get_state(self):
        state = self.stub.GetState(Empty())
        return proto_to_state(state)

    def get_nve_state(self):
        nve_state = self.stub.GetNVEState(Empty())
        return proto_to_nve_state(nve_state)

    def get_agent_state(self):
        agent_state = self.stub.GetAgentState(Empty())
        return proto_to_agent_state(agent_state)

    def get_object_state(self):
        object_state = self.stub.GetObjectState(Empty())
        return proto_to_object_state(object_state)

    def step(self):
        self.state = proto_to_state(self.stub.Step(Empty()))
        return self.state

    def is_started(self):
        return self.stub.IsStarted(Empty()).is_started
