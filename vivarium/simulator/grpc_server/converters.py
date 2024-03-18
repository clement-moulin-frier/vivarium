from jax_md.rigid_body import RigidBody

from vivarium.simulator.grpc_server.numproto.numproto import proto_to_ndarray, ndarray_to_proto
from vivarium.simulator.sim_computation import State, SimulatorState, NVEState, AgentState, ObjectState
import simulator_pb2


def proto_to_state(state):
    return State(simulator_state=proto_to_simulator_state(state.simulator_state),
                 nve_state=proto_to_nve_state(state.nve_state),
                 agent_state=proto_to_agent_state(state.agent_state),
                 object_state=proto_to_object_state(state.object_state))


def proto_to_simulator_state(simulator_state):
    return SimulatorState(idx=proto_to_ndarray(simulator_state.idx),
                          box_size=proto_to_ndarray(simulator_state.box_size),
                          n_agents=proto_to_ndarray(simulator_state.n_agents),
                          n_objects=proto_to_ndarray(simulator_state.n_objects),
                          num_steps_lax=proto_to_ndarray(simulator_state.num_steps_lax),
                          dt=proto_to_ndarray(simulator_state.dt),
                          freq=proto_to_ndarray(simulator_state.freq),
                          neighbor_radius=proto_to_ndarray(simulator_state.neighbor_radius),
                          to_jit=proto_to_ndarray(simulator_state.to_jit),
                          use_fori_loop=proto_to_ndarray(simulator_state.use_fori_loop),
                          collision_eps=proto_to_ndarray(simulator_state.collision_eps),
                          collision_alpha=proto_to_ndarray(simulator_state.collision_alpha)
                          )


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
                    friction=proto_to_ndarray(nve_state.friction).astype(float),
                    exists=proto_to_ndarray(nve_state.exists).astype(int)
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


def state_to_proto(state):
    return simulator_pb2.State(simulator_state=simulator_state_to_proto(state.simulator_state),
                               nve_state=nve_state_to_proto(state.nve_state),
                               agent_state=agent_state_to_proto(state.agent_state),
                               object_state=object_state_to_proto(state.object_state))


def simulator_state_to_proto(simulator_state):
    return simulator_pb2.SimulatorState(
        idx=ndarray_to_proto(simulator_state.idx),
        box_size=ndarray_to_proto(simulator_state.box_size),
        n_agents=ndarray_to_proto(simulator_state.n_agents),
        n_objects=ndarray_to_proto(simulator_state.n_objects),
        num_steps_lax=ndarray_to_proto(simulator_state.num_steps_lax),
        dt=ndarray_to_proto(simulator_state.dt),
        freq=ndarray_to_proto(simulator_state.freq),
        neighbor_radius=ndarray_to_proto(simulator_state.neighbor_radius),
        to_jit=ndarray_to_proto(simulator_state.to_jit),
        use_fori_loop=ndarray_to_proto(simulator_state.use_fori_loop),
        collision_eps=ndarray_to_proto(simulator_state.collision_eps),
        collision_alpha=ndarray_to_proto(simulator_state.collision_alpha)
    )


def nve_state_to_proto(nve_state):
    return simulator_pb2.NVEState(position=simulator_pb2.RigidBody(center=ndarray_to_proto(nve_state.position.center),
                                                                   orientation=ndarray_to_proto(nve_state.position.orientation)),
                                  momentum=simulator_pb2.RigidBody(center=ndarray_to_proto(nve_state.momentum.center),
                                                                   orientation=ndarray_to_proto(nve_state.momentum.orientation)),
                                  force=simulator_pb2.RigidBody(center=ndarray_to_proto(nve_state.force.center),
                                                                orientation=ndarray_to_proto(nve_state.force.orientation)),
                                  mass=simulator_pb2.RigidBody(center=ndarray_to_proto(nve_state.mass.center),
                                                               orientation=ndarray_to_proto(nve_state.mass.orientation)),
                                  entity_type=ndarray_to_proto(nve_state.entity_type),
                                  entity_idx=ndarray_to_proto(nve_state.entity_idx),
                                  diameter=ndarray_to_proto(nve_state.diameter),
                                  friction=ndarray_to_proto(nve_state.friction),
                                  exists=ndarray_to_proto(nve_state.exists)
                                  )


def agent_state_to_proto(agent_state):
    return simulator_pb2.AgentState(nve_idx=ndarray_to_proto(agent_state.nve_idx),
                                    prox=ndarray_to_proto(agent_state.prox),
                                    motor=ndarray_to_proto(agent_state.motor),
                                    behavior=ndarray_to_proto(agent_state.behavior),
                                    wheel_diameter=ndarray_to_proto(agent_state.wheel_diameter),
                                    speed_mul=ndarray_to_proto(agent_state.speed_mul),
                                    theta_mul=ndarray_to_proto(agent_state.theta_mul),
                                    proxs_dist_max=ndarray_to_proto(agent_state.proxs_dist_max),
                                    proxs_cos_min=ndarray_to_proto(agent_state.proxs_cos_min),
                                    color=ndarray_to_proto(agent_state.color),
                                    )


def object_state_to_proto(object_state):
    return simulator_pb2.ObjectState(nve_idx=ndarray_to_proto(object_state.nve_idx),
                                     color=ndarray_to_proto(object_state.color)
                                     )