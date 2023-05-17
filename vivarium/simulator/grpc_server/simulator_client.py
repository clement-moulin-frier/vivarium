import json

import grpc
from vivarium.simulator.grpc_server import simulator_pb2_grpc
import vivarium.simulator.grpc_server.simulator_pb2 as simulator_pb2

from vivarium.simulator.config import SimulatorConfig, AgentConfig
from vivarium.simulator.sim_computation import Population, NVEState
from vivarium.simulator.simulator_client_abc import SimulatorClient

from jax_md.rigid_body import RigidBody


from numproto.numproto import ndarray_to_proto, proto_to_ndarray

from protobuf_to_dict import protobuf_to_dict, dict_to_protobuf

import numpy as np

Empty = simulator_pb2.google_dot_protobuf_dot_empty__pb2.Empty


class SimulatorGRPCClient(SimulatorClient):
    def __init__(self, name=None):
        self.name = name
        channel = grpc.insecure_channel('localhost:50051')
        self.stub = simulator_pb2_grpc.SimulatorServerStub(channel)

    def start(self):
        self.stub.Start(Empty())

    def stop(self):
        self.stub.Stop(Empty())

    def get_change_time(self):
        return self.stub.GetChangeTime(Empty()).time

    def get_sim_config_dict(self):
        config = self.stub.GetSimulationConfig(Empty())
        return protobuf_to_dict(config)

    def get_sim_config(self):
        serialized = self.stub.GetSimulationConfigSerialized(Empty()).serialized
        # print('get_sim_config', serialized)
        return SimulatorConfig(**SimulatorConfig.param.deserialize_parameters(serialized))

    def get_recorded_changes(self):
        changes = self.stub.GetRecordedChanges(simulator_pb2.Name(name=self.name))
        d = json.loads(changes.serialized_dict)
        if changes.has_entity_behaviors:
            d['entity_behaviors'] = proto_to_ndarray(changes.entity_behaviors)
        return d

    def get_agent_config_dict(self):
        config = self.stub.GetAgentConfig(Empty())
        return protobuf_to_dict(config)

    def get_agent_config(self, idx):
        serialized = self.stub.GetAgentConfigSerialized(simulator_pb2.AgentIdx(idx=idx)).serialized
        return AgentConfig(**AgentConfig.param.deserialize_parameters(serialized))

    def get_agent_configs(self):
        agent_configs_message = self.stub.GetAgentConfigs(Empty()).agent_configs
        res = []
        for config in agent_configs_message:
            d = protobuf_to_dict(config)
            res.append(AgentConfig(**d))
        return config

    def get_population_config_dict(self):
        config = self.stub.GetPopulationConfig(Empty())
        return protobuf_to_dict(config)

    def get_population_config(self):
        serialized = self.stub.GetPopulationConfigSerialized(Empty()).serialized
        return PopulationConfig(**PopulationConfig.param.deserialize_parameters(serialized))

    def set_population_config(self, population_config):
        config = simulator_pb2.PopulationConfig(**population_config.to_dict())
        self.stub.SetPopulationConfig(config)

    def set_simulation_config(self, simulation_config):
        d = simulation_config.to_dict()
        print('set_simulation_config', d)
        # if 'entity_behaviors' in d:
        #     d['entity_behaviors'] = ndarray_to_proto(d['entity_behaviors'])
        config = simulator_pb2.SimulationConfig(**d)
        name = simulator_pb2.Name(name=self.name)
        config_sender_name = simulator_pb2.SimulationConfigSenderName(name=name, config=config)
        self.stub.SetSimulationConfig(config_sender_name)

    def set_agent_config(self, agent_idx, agent_config):
        d = agent_config.to_dict()
        print('set_agent_config', d)
        # if 'entity_behaviors' in d:
        #     d['entity_behaviors'] = ndarray_to_proto(d['entity_behaviors'])
        config = simulator_pb2.AgentConfig(**d)
        name = simulator_pb2.Name(name=self.name)
        idx = simulator_pb2.AgentIdx(idx=agent_idx)
        config_idx_sender_name = simulator_pb2.AgentConfigIdxSenderName(name=name, config=config, idx=idx)
        self.stub.SetAgentConfig(config_idx_sender_name)

    def set_simulation_config_serialized(self, simulation_config):
        serialized = simulation_config.param.serialize_parameters(subset=simulation_config.export_fields)
        # print('set_simulation_config', serialized)
        self.stub.SetSimulationConfigSerialized(simulator_pb2.SimulationConfigSerialized(serialized=serialized))

    def set_motors(self, agent_idx, motors):
        if type(agent_idx) is int:
            start, stop, step = agent_idx, agent_idx + 1, 1
        else:
            start, stop, step = agent_idx.start, agent_idx.stop, agent_idx.step
        motors = simulator_pb2.Motors(agent_slice=simulator_pb2.Slice(start=start, stop=stop, step=step),
                                      motors=ndarray_to_proto(motors))
        self.stub.SetMotors(motors)

    def set_behaviors(self, agent_idx, behavior):
        if type(agent_idx) is int:
            start, stop, step = agent_idx, agent_idx + 1, 1
        else:
            start, stop, step = agent_idx.start, agent_idx.stop, agent_idx.step
        behavior = simulator_pb2.Behaviors(agent_slice=simulator_pb2.Slice(start=start, stop=stop, step=step),
                                           behavior=behavior)
        self.stub.SetBehaviors(behavior)

    def get_state(self):
        return self.stub.GetStateMessage(Empty())

    def get_state_arrays(self):
        state = self.stub.GetStateArrays(Empty())
        return Population(position=proto_to_ndarray(state.positions),
                          theta=proto_to_ndarray(state.thetas),
                          prox=proto_to_ndarray(state.proxs),
                          motor=proto_to_ndarray(state.motors),
                          entity_type=state.entity_type)

    def get_nve_state(self):
        state = self.stub.GetNVEState(Empty())
        return NVEState(position=RigidBody(center=proto_to_ndarray(state.position.center),
                                           orientation=proto_to_ndarray(state.position.orientation)),
                        momentum=RigidBody(center=proto_to_ndarray(state.momentum.center),
                                           orientation=proto_to_ndarray(state.momentum.orientation)),
                        force=RigidBody(center=proto_to_ndarray(state.force.center),
                                        orientation=proto_to_ndarray(state.force.orientation)),
                        mass=RigidBody(center=proto_to_ndarray(state.mass.center),
                                       orientation=proto_to_ndarray(state.mass.orientation)),
                        prox=proto_to_ndarray(state.prox),
                        motor=proto_to_ndarray(state.motor),
                        behavior=proto_to_ndarray(state.behavior),
                        wheel_diameter=proto_to_ndarray(state.wheel_diameter),
                        base_length=proto_to_ndarray(state.base_length),
                        speed_mul=proto_to_ndarray(state.speed_mul),
                        theta_mul=proto_to_ndarray(state.theta_mul),
                        proxs_dist_max=proto_to_ndarray(state.proxs_dist_max),
                        proxs_cos_min=proto_to_ndarray(state.proxs_cos_min),
                        entity_type=proto_to_ndarray(state.entity_type)
                        )

    def is_started(self):
        return self.stub.IsStarted(Empty()).is_started

    def update_neighbor_fn(self):
        self.stub.UpdateNeighborFn(Empty())

    def update_state_neighbors(self):
        self.stub.UpdateStateNeighbors(Empty())

    def update_function_update(self):
        self.stub.UpdateFunctionUpdate(Empty())

    def update_behaviors(self):
        self.stub.UpdateBehaviors(Empty())
