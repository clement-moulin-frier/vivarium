# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
import simulator_pb2 as simulator__pb2


class SimulatorServerStub(object):
    """Interface exported by the server.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Step = channel.unary_unary(
                '/simulator.SimulatorServer/Step',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=simulator__pb2.State.FromString,
                )
        self.GetState = channel.unary_unary(
                '/simulator.SimulatorServer/GetState',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=simulator__pb2.State.FromString,
                )
        self.GetNVEState = channel.unary_unary(
                '/simulator.SimulatorServer/GetNVEState',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=simulator__pb2.EntityState.FromString,
                )
        self.GetAgentState = channel.unary_unary(
                '/simulator.SimulatorServer/GetAgentState',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=simulator__pb2.AgentState.FromString,
                )
        self.GetObjectState = channel.unary_unary(
                '/simulator.SimulatorServer/GetObjectState',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=simulator__pb2.ObjectState.FromString,
                )
        self.SetState = channel.unary_unary(
                '/simulator.SimulatorServer/SetState',
                request_serializer=simulator__pb2.StateChange.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )
        self.GetSceneName = channel.unary_unary(
                '/simulator.SimulatorServer/GetSceneName',
                request_serializer=simulator__pb2.Scene.SerializeToString,
                response_deserializer=simulator__pb2.Scene.FromString,
                )
        self.GetSubtypesLabels = channel.unary_unary(
                '/simulator.SimulatorServer/GetSubtypesLabels',
                request_serializer=simulator__pb2.SubtypesLabels.SerializeToString,
                response_deserializer=simulator__pb2.SubtypesLabels.FromString,
                )
        self.IsStarted = channel.unary_unary(
                '/simulator.SimulatorServer/IsStarted',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=simulator__pb2.IsStartedState.FromString,
                )
        self.Start = channel.unary_unary(
                '/simulator.SimulatorServer/Start',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )
        self.Stop = channel.unary_unary(
                '/simulator.SimulatorServer/Stop',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )


class SimulatorServerServicer(object):
    """Interface exported by the server.
    """

    def Step(self, request, context):
        """Do a step in the simulation
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetState(self, request, context):
        """Get one of the states of the simulation
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetNVEState(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetAgentState(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetObjectState(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetState(self, request, context):
        """set the state of the simulation
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetSceneName(self, request, context):
        """send the labels of entities
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetSubtypesLabels(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def IsStarted(self, request, context):
        """Handle the connection between the server and the clients
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Start(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Stop(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SimulatorServerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Step': grpc.unary_unary_rpc_method_handler(
                    servicer.Step,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=simulator__pb2.State.SerializeToString,
            ),
            'GetState': grpc.unary_unary_rpc_method_handler(
                    servicer.GetState,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=simulator__pb2.State.SerializeToString,
            ),
            'GetNVEState': grpc.unary_unary_rpc_method_handler(
                    servicer.GetNVEState,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=simulator__pb2.EntityState.SerializeToString,
            ),
            'GetAgentState': grpc.unary_unary_rpc_method_handler(
                    servicer.GetAgentState,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=simulator__pb2.AgentState.SerializeToString,
            ),
            'GetObjectState': grpc.unary_unary_rpc_method_handler(
                    servicer.GetObjectState,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=simulator__pb2.ObjectState.SerializeToString,
            ),
            'SetState': grpc.unary_unary_rpc_method_handler(
                    servicer.SetState,
                    request_deserializer=simulator__pb2.StateChange.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
            'GetSceneName': grpc.unary_unary_rpc_method_handler(
                    servicer.GetSceneName,
                    request_deserializer=simulator__pb2.Scene.FromString,
                    response_serializer=simulator__pb2.Scene.SerializeToString,
            ),
            'GetSubtypesLabels': grpc.unary_unary_rpc_method_handler(
                    servicer.GetSubtypesLabels,
                    request_deserializer=simulator__pb2.SubtypesLabels.FromString,
                    response_serializer=simulator__pb2.SubtypesLabels.SerializeToString,
            ),
            'IsStarted': grpc.unary_unary_rpc_method_handler(
                    servicer.IsStarted,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=simulator__pb2.IsStartedState.SerializeToString,
            ),
            'Start': grpc.unary_unary_rpc_method_handler(
                    servicer.Start,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
            'Stop': grpc.unary_unary_rpc_method_handler(
                    servicer.Stop,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'simulator.SimulatorServer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class SimulatorServer(object):
    """Interface exported by the server.
    """

    @staticmethod
    def Step(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulator.SimulatorServer/Step',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            simulator__pb2.State.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetState(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulator.SimulatorServer/GetState',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            simulator__pb2.State.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetNVEState(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulator.SimulatorServer/GetNVEState',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            simulator__pb2.EntityState.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAgentState(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulator.SimulatorServer/GetAgentState',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            simulator__pb2.AgentState.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetObjectState(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulator.SimulatorServer/GetObjectState',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            simulator__pb2.ObjectState.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetState(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulator.SimulatorServer/SetState',
            simulator__pb2.StateChange.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetSceneName(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulator.SimulatorServer/GetSceneName',
            simulator__pb2.Scene.SerializeToString,
            simulator__pb2.Scene.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetSubtypesLabels(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulator.SimulatorServer/GetSubtypesLabels',
            simulator__pb2.SubtypesLabels.SerializeToString,
            simulator__pb2.SubtypesLabels.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def IsStarted(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulator.SimulatorServer/IsStarted',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            simulator__pb2.IsStartedState.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Start(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulator.SimulatorServer/Start',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Stop(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/simulator.SimulatorServer/Stop',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
