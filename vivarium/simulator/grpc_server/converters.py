from jax_md.rigid_body import RigidBody

import simulator_pb2

from vivarium.simulator.grpc_server.numproto.numproto import proto_to_ndarray, ndarray_to_proto
from vivarium.simulator.states import State, SimulatorState, EntitiesState, AgentState, ObjectState


def proto_to_state(state):
    return State(simulator_state=proto_to_simulator_state(state.simulator_state),
                 entities_state=proto_to_nve_state(state.entities_state),
                 agent_state=proto_to_agent_state(state.agent_state),
                 object_state=proto_to_object_state(state.object_state))


def proto_to_simulator_state(simulator_state):
    return SimulatorState(idx=proto_to_ndarray(simulator_state.idx).astype(int),
                          box_size=proto_to_ndarray(simulator_state.box_size).astype(float),
                          max_agents=proto_to_ndarray(simulator_state.max_agents).astype(int),
                          max_objects=proto_to_ndarray(simulator_state.max_objects).astype(int),
                          num_steps_lax=proto_to_ndarray(simulator_state.num_steps_lax).astype(int),
                          dt=proto_to_ndarray(simulator_state.dt).astype(float),
                          freq=proto_to_ndarray(simulator_state.freq).astype(float),
                          neighbor_radius=proto_to_ndarray(simulator_state.neighbor_radius).astype(float),
                          to_jit=proto_to_ndarray(simulator_state.to_jit).astype(int),
                          use_fori_loop=proto_to_ndarray(simulator_state.use_fori_loop).astype(int)
                          )


def proto_to_nve_state(entities_state):
    return EntitiesState(position=RigidBody(center=proto_to_ndarray(entities_state.position.center).astype(float),
                                       orientation=proto_to_ndarray(entities_state.position.orientation).astype(float)),
                    momentum=RigidBody(center=proto_to_ndarray(entities_state.momentum.center).astype(float),
                                       orientation=proto_to_ndarray(entities_state.momentum.orientation).astype(float)),
                    force=RigidBody(center=proto_to_ndarray(entities_state.force.center).astype(float),
                                    orientation=proto_to_ndarray(entities_state.force.orientation).astype(float)),
                    mass=RigidBody(center=proto_to_ndarray(entities_state.mass.center).astype(float),
                                   orientation=proto_to_ndarray(entities_state.mass.orientation).astype(float)),
                    entity_type=proto_to_ndarray(entities_state.entity_type).astype(int),
                    entity_idx=proto_to_ndarray(entities_state.entity_idx).astype(int),
                    diameter=proto_to_ndarray(entities_state.diameter).astype(float),
                    friction=proto_to_ndarray(entities_state.friction).astype(float),
                    exists=proto_to_ndarray(entities_state.exists).astype(int),
                    collision_eps=proto_to_ndarray(entities_state.collision_eps).astype(float),
                    collision_alpha=proto_to_ndarray(entities_state.collision_alpha).astype(float)
                    )


def proto_to_agent_state(agent_state):
    return AgentState(nve_idx=proto_to_ndarray(agent_state.nve_idx).astype(int),
                      prox=proto_to_ndarray(agent_state.prox).astype(float),
                      motor=proto_to_ndarray(agent_state.motor).astype(float),
                      behavior=proto_to_ndarray(agent_state.behavior).astype(int),
                      wheel_diameter=proto_to_ndarray(agent_state.wheel_diameter).astype(float),
                      speed_mul=proto_to_ndarray(agent_state.speed_mul).astype(float),
                      max_speed=proto_to_ndarray(agent_state.max_speed).astype(float),
                      theta_mul=proto_to_ndarray(agent_state.theta_mul).astype(float),
                      proxs_dist_max=proto_to_ndarray(agent_state.proxs_dist_max).astype(float),
                      proxs_cos_min=proto_to_ndarray(agent_state.proxs_cos_min).astype(float),
                      color=proto_to_ndarray(agent_state.color).astype(float),
                      )


def proto_to_object_state(object_state):
    return ObjectState(nve_idx=proto_to_ndarray(object_state.nve_idx).astype(int),
                       color=proto_to_ndarray(object_state.color).astype(float),
                       )


def state_to_proto(state):
    return simulator_pb2.State(simulator_state=simulator_state_to_proto(state.simulator_state),
                               entities_state=nve_state_to_proto(state.entities_state),
                               agent_state=agent_state_to_proto(state.agent_state),
                               object_state=object_state_to_proto(state.object_state))


def simulator_state_to_proto(simulator_state):
    return simulator_pb2.SimulatorState(
        idx=ndarray_to_proto(simulator_state.idx),
        box_size=ndarray_to_proto(simulator_state.box_size),
        max_agents=ndarray_to_proto(simulator_state.max_agents),
        max_objects=ndarray_to_proto(simulator_state.max_objects),
        num_steps_lax=ndarray_to_proto(simulator_state.num_steps_lax),
        dt=ndarray_to_proto(simulator_state.dt),
        freq=ndarray_to_proto(simulator_state.freq),
        neighbor_radius=ndarray_to_proto(simulator_state.neighbor_radius),
        to_jit=ndarray_to_proto(simulator_state.to_jit),
        use_fori_loop=ndarray_to_proto(simulator_state.use_fori_loop)
    )


def nve_state_to_proto(entities_state):
    return simulator_pb2.EntitiesState(position=simulator_pb2.RigidBody(center=ndarray_to_proto(entities_state.position.center),
                                                                   orientation=ndarray_to_proto(entities_state.position.orientation)),
                                  momentum=simulator_pb2.RigidBody(center=ndarray_to_proto(entities_state.momentum.center),
                                                                   orientation=ndarray_to_proto(entities_state.momentum.orientation)),
                                  force=simulator_pb2.RigidBody(center=ndarray_to_proto(entities_state.force.center),
                                                                orientation=ndarray_to_proto(entities_state.force.orientation)),
                                  mass=simulator_pb2.RigidBody(center=ndarray_to_proto(entities_state.mass.center),
                                                               orientation=ndarray_to_proto(entities_state.mass.orientation)),
                                  entity_type=ndarray_to_proto(entities_state.entity_type),
                                  entity_idx=ndarray_to_proto(entities_state.entity_idx),
                                  diameter=ndarray_to_proto(entities_state.diameter),
                                  friction=ndarray_to_proto(entities_state.friction),
                                  exists=ndarray_to_proto(entities_state.exists),
                                  collision_eps=ndarray_to_proto(entities_state.collision_eps),
                                  collision_alpha=ndarray_to_proto(entities_state.collision_alpha)
                                  )


def agent_state_to_proto(agent_state):
    return simulator_pb2.AgentState(nve_idx=ndarray_to_proto(agent_state.nve_idx),
                                    prox=ndarray_to_proto(agent_state.prox),
                                    motor=ndarray_to_proto(agent_state.motor),
                                    behavior=ndarray_to_proto(agent_state.behavior),
                                    wheel_diameter=ndarray_to_proto(agent_state.wheel_diameter),
                                    speed_mul=ndarray_to_proto(agent_state.speed_mul),
                                    max_speed=ndarray_to_proto(agent_state.max_speed),
                                    theta_mul=ndarray_to_proto(agent_state.theta_mul),
                                    proxs_dist_max=ndarray_to_proto(agent_state.proxs_dist_max),
                                    proxs_cos_min=ndarray_to_proto(agent_state.proxs_cos_min),
                                    color=ndarray_to_proto(agent_state.color),
                                    )


def object_state_to_proto(object_state):
    return simulator_pb2.ObjectState(nve_idx=ndarray_to_proto(object_state.nve_idx),
                                     color=ndarray_to_proto(object_state.color)
                                     )