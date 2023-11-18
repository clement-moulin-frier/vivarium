from jax_md.rigid_body import RigidBody

from vivarium.simulator.grpc_server.numproto.numproto import proto_to_ndarray
from vivarium.simulator.sim_computation import State, NVEState, AgentState, ObjectState


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
