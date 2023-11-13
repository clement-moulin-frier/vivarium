import json

import grpc
from vivarium.simulator.grpc_server import simulator_pb2_grpc
import vivarium.simulator.grpc_server.simulator_pb2 as simulator_pb2

from vivarium.simulator.config import SimulatorConfig, AgentConfig
from vivarium.simulator.sim_computation import NVEState, AgentState, ObjectState, State
from vivarium.simulator.simulator_client_abc import SimulatorClient

from jax_md.rigid_body import RigidBody

from numproto.numproto import ndarray_to_proto, proto_to_ndarray

import dill
import threading
from contextlib import contextmanager

Empty = simulator_pb2.google_dot_protobuf_dot_empty__pb2.Empty


def proto_to_state(state):
    return State(nve_state=proto_to_nve_state(state.nve_state),
                 agent_state=proto_to_agent_state(state.agent_state),
                 object_state=proto_to_object_state(state.object_state))


def proto_to_nve_state(nve_state):
    return NVEState(position=RigidBody(center=proto_to_ndarray(nve_state.position.center).astype(float),
                                       orientation=proto_to_ndarray(nve_state.position.orientation).astype(float)),
                    momentum=RigidBody(center=proto_to_ndarray(nve_state.momentum.center).astype(float),
                                       orientation=proto_to_ndarray(nve_state.momentum.orientation).astype(float)),
                    force=RigidBody(center=proto_to_ndarray(nve_state.force.center).astype(float),
                                    orientation=proto_to_ndarray(nve_state.force.orientation).astype(float)),
                    mass=RigidBody(center=proto_to_ndarray(nve_state.mass.center).astype(float),
                                   orientation=proto_to_ndarray(nve_state.mass.orientation).astype(float)),
                    entity_type=proto_to_ndarray(nve_state.entity_type).astype(int),
                    entity_idx=proto_to_ndarray(nve_state.entity_idx).astype(int),
                    diameter=proto_to_ndarray(nve_state.diameter).astype(float),
                    friction=proto_to_ndarray(nve_state.friction).astype(float)
                    )


def proto_to_agent_state(agent_state):
    return AgentState(nve_idx=proto_to_ndarray(agent_state.nve_idx).astype(int),
                      prox=proto_to_ndarray(agent_state.prox).astype(float),
                      motor=proto_to_ndarray(agent_state.motor).astype(float),
                      behavior=proto_to_ndarray(agent_state.behavior).astype(int),
                      wheel_diameter=proto_to_ndarray(agent_state.wheel_diameter).astype(float),
                      speed_mul=proto_to_ndarray(agent_state.speed_mul).astype(float),
                      theta_mul=proto_to_ndarray(agent_state.theta_mul).astype(float),
                      proxs_dist_max=proto_to_ndarray(agent_state.proxs_dist_max).astype(float),
                      proxs_cos_min=proto_to_ndarray(agent_state.proxs_cos_min).astype(float),
                      color=proto_to_ndarray(agent_state.color).astype(float),
                      )


def proto_to_object_state(object_state):
    return ObjectState(nve_idx=proto_to_ndarray(object_state.nve_idx).astype(int),
                       color=proto_to_ndarray(object_state.color).astype(float),
                       )


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

    def get_sim_config(self):
        serialized = self.stub.GetSimulationConfigSerialized(Empty()).serialized
        return SimulatorConfig(**SimulatorConfig.param.deserialize_parameters(serialized))

    def get_recorded_changes(self):
        changes = self.stub.GetRecordedChanges(simulator_pb2.Name(name=self.name))
        d = json.loads(changes.serialized_dict)
        if changes.has_entity_behaviors:
            d['entity_behaviors'] = proto_to_ndarray(changes.entity_behaviors)
        return d

    def get_agent_config(self, idx):
        print('get_agent_config')
        serialized = self.stub.GetAgentConfigSerialized(simulator_pb2.AgentIdx(idx=idx)).serialized
        return AgentConfig(**AgentConfig.param.deserialize_parameters(serialized))

    def get_agent_configs(self):
        serialized = self.stub.GetAgentConfigsSerialized(Empty()).serialized
        return [AgentConfig(**AgentConfig.param.deserialize_parameters(s)) for s in serialized]

    def set_simulation_config(self, simulation_config_dict):
        print('set_simulation_config', simulation_config_dict)
        serial_dict = json.dumps(simulation_config_dict)
        serial_dict = simulator_pb2.SerializedDict(serialized=serial_dict)
        name = simulator_pb2.Name(name=self.name)
        dict_sender_name = simulator_pb2.SerializedDictSenderName(name=name, dict=serial_dict)
        self.stub.SetSimulationConfig(dict_sender_name)

    def set_agent_config(self, selected_agents, agent_config_dict):
        print('set_agent_config', agent_config_dict)
        serial_dict = json.dumps(agent_config_dict)
        serial_dict = simulator_pb2.SerializedDict(serialized=serial_dict)
        name = simulator_pb2.Name(name=self.name)
        idx = simulator_pb2.AgentIdx(idx=selected_agents)
        dict_idx_sender_name = simulator_pb2.SerializedDictIdxSenderName(name=name, dict=serial_dict, idx=idx)
        self.stub.SetAgentConfig(dict_idx_sender_name)

    def set_motors(self, selected_agents, motor_idx, value):
        agent_idx = simulator_pb2.AgentIdx(idx=selected_agents)
        motor_info = simulator_pb2.MotorInfo(agent_idx=agent_idx, motor_idx=motor_idx, value=value)
        self.stub.SetMotors(motor_info)

    def set_state(self, nested_field, nve_idx, column_idx, value):
        state_change = simulator_pb2.StateChange(nested_field=nested_field, nve_idx=nve_idx, col_idx=column_idx,
                                                 value=ndarray_to_proto(value))
        self.stub.SetState(state_change)

    def set_simulation_config_serialized(self, simulation_config):
        serialized = simulation_config.param.serialize_parameters(subset=simulation_config.param_names())
        self.stub.SetSimulationConfigSerialized(simulator_pb2.SimulationConfigSerialized(serialized=serialized))

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

    def start_behavior(self, agent_idx, behavior_fn):
        behavior = simulator_pb2.Behavior(agent_idx=agent_idx,
                                          function=dill.dumps(behavior_fn))
        self.stub.StartBehavior(behavior)

    def agent_step(self, agent_idx, motor):
        prox = self.stub.AgentStep(simulator_pb2.Motor(agent_idx=agent_idx, motor=motor))
        return prox.prox

    def step(self):
        self.state = proto_to_state(self.stub.Step(Empty()))
        return self.state

    def _start_streaming(self):
        self.streaming_started = True
        for state in self.stub.NVEStateStream(Empty()):
            if not self.streaming_started:
                print('Stop streaming')
                break
            self.state = state_proto_to_nve_state(state)
    def start_streaming(self):
        print('Start streaming')
        threading.Thread(target=self._start_streaming).start()

    def stop_streaming(self):
        self.stub.StopNVEStream()


    # def start_behavior(self, agent_idx, behavior_fn):
    #     class BehaviorIterator:
    #         def __init__(self, behavior_fn, init_prox):
    #             self.behavior_fn = behavior_fn
    #             self.prox = init_prox
    #         def __iter__(self):
    #             motor = simulator_pb2.Motor(agent_idx=agent_idx, motor=self.behavior_fn(self.prox))
    #             return motor
    #
    #         def __next__(self):
    #             motor = simulator_pb2.Motor(agent_idx=agent_idx, motor=self.behavior_fn(self.prox))
    #             return motor
    #
    #     behavior_iterator = BehaviorIterator(behavior_fn, self.get_nve_state().prox[agent_idx])
    #
    #     for prox in self.stub.SensoryMotorStream(behavior_iterator):
    #         behavior_iterator.prox = prox.prox

    def add_agents(self, n_agents, agent_config):
        d = agent_config.param.serialize_parameters(subset=agent_config.param_names())
        # config = simulator_pb2.AgentConfig(**agent_config.to_dict())
        input = simulator_pb2.AddAgentInput(n_agents=n_agents,
                                            serialized_config=d)
        return self.stub.AddAgents(input)

    def remove_agents(self, agent_idx):
        self.stub.RemoveAgents(simulator_pb2.AgentIdx(idx=agent_idx))

    def is_started(self):
        return self.stub.IsStarted(Empty()).is_started

