import jax
from jax import ops, vmap, lax
import jax.numpy as jnp

from jax_md import space, rigid_body, util, simulate, energy, quantity
from jax_md.dataclasses import dataclass

from functools import partial
from enum import Enum
from itertools import compress


f32 = util.f32

SPACE_NDIMS = 2

class EntityType(Enum):
    AGENT = 0
    OBJECT = 1

    def to_state_type(self):
        return StateType(self.value)

class StateType(Enum):
    AGENT = 0
    OBJECT = 1
    SIMULATOR = 2

    def is_entity(self):
        return self != StateType.SIMULATOR

    def to_entity_type(self):
        assert self.is_entity()
        return EntityType(self.value)



@dataclass
class NVEState(simulate.NVEState):
    entity_type: util.Array
    entity_idx: util.Array  # idx in XState (e.g. AgentState)
    diameter: util.Array
    friction: util.Array
    exists: util.Array

    @property
    def velocity(self) -> util.Array:
        return self.momentum / self.mass

@dataclass
class AgentState:
    nve_idx: util.Array  # idx in NVEState
    prox: util.Array
    motor: util.Array
    behavior: util.Array
    wheel_diameter: util.Array
    speed_mul: util.Array
    theta_mul: util.Array
    proxs_dist_max: util.Array
    proxs_cos_min: util.Array
    color: util.Array


@dataclass
class ObjectState:
    nve_idx: util.Array  # idx in NVEState
    color: util.Array

@dataclass
class SimulatorState:
    idx: util.Array
    box_size: util.Array
    n_agents: util.Array
    n_objects: util.Array
    num_steps_lax: util.Array
    dt: util.Array
    freq: util.Array
    neighbor_radius: util.Array
    to_jit: util.Array
    use_fori_loop: util.Array

    @staticmethod
    def get_type(attr):
        if attr in ['idx', 'n_agents', 'n_objects', 'num_steps_lax']:
            return int
        elif attr in ['box_size', 'dt', 'freq', 'neighbor_radius']:
            return float
        elif attr in ['to_jit', 'use_fori_loop']:
            return bool
        else:
            raise ValueError()

@dataclass
class State:
    simulator_state: SimulatorState
    nve_state: NVEState
    agent_state: AgentState
    object_state: ObjectState

    def field(self, stype_or_nested_fields):
        if isinstance(stype_or_nested_fields, StateType):
            name = stype_or_nested_fields.name.lower()
            nested_fields = (f'{name}_state', )
        else:
            nested_fields = stype_or_nested_fields

        res = self
        for f in nested_fields:
            res = getattr(res, f)

        return res

    def nve_idx(self, etype):
        cond = self.e_cond(etype)
        return compress(range(len(cond)), cond)  # https://stackoverflow.com/questions/21448225/getting-indices-of-true-values-in-a-boolean-list

    def nve_idx(self, etype, entity_idx):
        return self.field(etype).nve_idx[entity_idx]

    def e_idx(self, etype):
        return self.nve_state.entity_idx[self.nve_state.entity_type == etype.value]

    def e_cond(self, etype):
        return self.nve_state.entity_type == etype.value

    def row_idx(self, field, nve_idx):
        return nve_idx if field == 'nve_state' else self.nve_state.entity_idx[jnp.array(nve_idx)]

    def __getattr__(self, name):
        def wrapper(e_type):
            value = getattr(self.nve_state, name)
            if isinstance(value, rigid_body.RigidBody):
                return rigid_body.RigidBody(center=value.center[self.e_cond(e_type)],
                                            orientation=value.orientation[self.e_cond(e_type)])
            else:
                return value[self.e_cond(e_type)]
        return wrapper


def normal(theta):
  return jnp.array([jnp.cos(theta), jnp.sin(theta)])

normal = vmap(normal)

def switch_fn(fn_list):
    def switch(index, *operands):
        return jax.lax.switch(index, fn_list, *operands)
    return switch


""" Helper functions for collisions """

def collision_energy(displacement_fn, r_a, r_b, l_a, l_b, epsilon, alpha):
    dist = jnp.linalg.norm(displacement_fn(r_a, r_b))
    sigma = l_a + l_b
    return energy.soft_sphere(dist, sigma=sigma, epsilon=epsilon, alpha=alpha)

collision_energy = vmap(collision_energy, (None, 0, 0, 0, 0, None, None))


def total_collision_energy(positions, diameter, neighbor, displacement, exists_mask, epsilon=1e-2, alpha=2, **kwargs):
    diameter = lax.stop_gradient(diameter)
    senders, receivers = neighbor.idx
    Ra = positions[senders]
    Rb = positions[receivers]
    l_a = diameter[senders]
    l_b = diameter[receivers]
    e = collision_energy(displacement, Ra, Rb, l_a, l_b, epsilon, alpha)
    # Set collision energy to zero if the sender or receiver is non existing
    e = jnp.where(exists_mask[senders] * exists_mask[receivers], e, 0.) 
    return jnp.sum(e)


""" Helper functions for motor function """

def lr_2_fwd_rot(left_spd, right_spd, base_length, wheel_diameter):
    fwd = (wheel_diameter / 4.) * (left_spd + right_spd)
    rot = 0.5 * (wheel_diameter / base_length) * (right_spd - left_spd)
    return fwd, rot


def fwd_rot_2_lr(fwd, rot, base_length, wheel_diameter):
    left = ((2.0 * fwd) - (rot * base_length)) / wheel_diameter
    right = ((2.0 * fwd) + (rot * base_length)) / wheel_diameter
    return left, right


def motor_command(wheel_activation, base_length, wheel_diameter):
  fwd, rot = lr_2_fwd_rot(wheel_activation[0], wheel_activation[1], base_length, wheel_diameter)
  return fwd, rot

motor_command = vmap(motor_command, (0, 0, 0))


""" Functions to compute the verlet force on the whole system"""

def get_verlet_force_fn(displacement):
    coll_force_fn = quantity.force(partial(total_collision_energy, displacement=displacement,
                                           epsilon=10., alpha=12))

    def collision_force(nve_state, neighbor, exists_mask):
        return coll_force_fn(nve_state.position.center, neighbor=neighbor, exists_mask=exists_mask, diameter=nve_state.diameter)

    def friction_force(nve_state, exists_mask):
        cur_vel = nve_state.momentum.center / nve_state.mass.center
        # stack the mask to give it the same shape as cur_vel (that has 2 rows for forward and angular velocities) 
        # mask = jnp.stack([exists_mask] * 2, axis=1) 
        # cur_vel = jnp.where(mask, cur_vel, 0.)
        return - jnp.tile(nve_state.friction, (SPACE_NDIMS, 1)).T * cur_vel

    def motor_force(state, exists_mask):
        agent_idx = state.agent_state.nve_idx
        body = rigid_body.RigidBody(center=state.nve_state.position.center[agent_idx],
                                    orientation=state.nve_state.position.orientation[agent_idx])
        fwd, rot = motor_command(state.agent_state.motor,
                                 state.nve_state.diameter[agent_idx],
                                 state.agent_state.wheel_diameter)
        n = normal(body.orientation)

        cur_vel = state.nve_state.momentum.center[agent_idx] / state.nve_state.mass.center[agent_idx]
        cur_fwd_vel = vmap(jnp.dot)(cur_vel, n)
        cur_rot_vel = state.nve_state.momentum.orientation[agent_idx] / state.nve_state.mass.orientation[agent_idx]
        fwd_delta = fwd - cur_fwd_vel
        rot_delta = rot - cur_rot_vel
        fwd_force = n * jnp.tile(fwd_delta, (SPACE_NDIMS, 1)).T * jnp.tile(state.agent_state.speed_mul, (SPACE_NDIMS, 1)).T
        rot_force = rot_delta * state.agent_state.theta_mul

        center=jnp.zeros_like(state.nve_state.position.center).at[agent_idx].set(fwd_force)
        orientation=jnp.zeros_like(state.nve_state.position.orientation).at[agent_idx].set(rot_force)

        # apply mask to make non existing agents stand still 
        orientation = jnp.where(exists_mask, orientation, 0.)
        # Because position has SPACE_NDMS dims, need to stack the mask to give it the same shape as center 
        exists_mask = jnp.stack([exists_mask] * SPACE_NDIMS, axis=1) 
        center = jnp.where(exists_mask, center, 0.)

        return rigid_body.RigidBody(center=center,
                                    orientation=orientation)
    

    def force_fn(state, neighbor, exists_mask):
        mf = motor_force(state, exists_mask)
        center = collision_force(state.nve_state, neighbor, exists_mask) + friction_force(state.nve_state, exists_mask) + mf.center
        orientation = mf.orientation
        return rigid_body.RigidBody(center=center, orientation=orientation)

    return force_fn


""" Helper functions for sensors """

def dist_theta(displ, theta):
    """
    Compute the relative distance and angle from a source agent to a target agent
    :param displ: Displacement vector (jnp arrray with shape (2,) from source to target
    :param theta: Orientation of the source agent (in the reference frame of the map)
    :return: dist: distance from source to target.
    relative_theta: relative angle of the target in the reference frame of the source agent (front direction at angle 0)
    """
    dist = jnp.linalg.norm(displ)
    norm_displ = displ / dist
    theta_displ = jnp.arccos(norm_displ[0]) * jnp.sign(jnp.arcsin(norm_displ[1]))
    relative_theta = theta_displ - theta
    return dist, relative_theta

proximity_map = vmap(dist_theta, (0, 0))


def sensor_fn(dist, relative_theta, dist_max, cos_min, target_exists):
    """
    Compute the proximeter activations (left, right) induced by the presence of an entity
    :param dist: distance from the agent to the entity
    :param relative_theta: angle of the entity in the reference frame of the agent (front direction at angle 0)
    :param dist_max: Max distance of the proximiter (will return 0. above this distance)
    :param cos_min: Field of view as a cosinus (e.g. cos_min = 0 means a pi/4 FoV on each proximeter, so pi/2 in total)
    :return: left and right proximeter activation in a jnp array with shape (2,)
    """
    cos_dir = jnp.cos(relative_theta)
    prox = 1. - (dist / dist_max)
    in_view = jnp.logical_and(dist < dist_max, cos_dir > cos_min)
    at_left = jnp.logical_and(True, jnp.sin(relative_theta) >= 0)
    left = in_view * at_left * prox
    right = in_view * (1. - at_left) * prox
    return jnp.array([left, right]) * target_exists  # i.e. 0 if target does not exist

sensor_fn = vmap(sensor_fn, (0, 0, 0, 0, 0))


def sensor(displ, theta, dist_max, cos_min, n_agents, senders, target_exists):
    dist, relative_theta = proximity_map(displ, theta)
    proxs = ops.segment_max(sensor_fn(dist, relative_theta, dist_max, cos_min, target_exists),
                            senders, n_agents)
    return proxs


""" Functions to compute the dynamics of the whole system """

def dynamics_rigid(displacement, shift, behavior_bank, force_fn=None):
    force_fn = force_fn or get_verlet_force_fn(displacement)
    multi_switch = jax.vmap(switch_fn(behavior_bank), (0, 0, 0))
    # shape = rigid_body.monomer

    def init_fn(state, key, kT=0.):
        key, subkey = jax.random.split(key)
        assert state.nve_state.momentum is None
        assert not jnp.any(state.nve_state.force.center) and not jnp.any(state.nve_state.force.orientation)

        state = state.set(nve_state=simulate.initialize_momenta(state.nve_state, key, kT))
        return state

    def physics_fn(state, force, shift_fn, dt, neighbor):
        """Apply a single step of velocity Verlet integration to a state."""
        # dt = f32(dt)
        dt_2 = dt / 2.  # f32(dt / 2)
        # state = sensorimotor(state, neighbor)  # now in step_fn
        nve_state = simulate.momentum_step(state.nve_state, dt_2)
        nve_state = simulate.position_step(nve_state, shift_fn, dt, neighbor=neighbor)
        nve_state = nve_state.set(force=force)
        nve_state = simulate.momentum_step(nve_state, dt_2)

        return state.set(nve_state=nve_state)

    def compute_prox(state, agent_neighs_idx, target_exists_mask):
        """
        Set agents' proximeter activations
        :param state: full simulation State
        :param agent_neighs_idx: Neighbor representation, where sources are only agents. Matrix of shape (2, n_pairs),
        where n_pairs is the number of neighbor entity pairs where sources (first row) are agent indexes.
        :param target_exists_mask: Specify which target entities exist. Vector with shape (n_entities,).
        target_exists_mask[i] is True (resp. False) if entity of index i in state.nve_state exists (resp. don't exist).
        :return:
        """
        body = state.nve_state.position
        mask = target_exists_mask[agent_neighs_idx[1, :]]
        senders, receivers = agent_neighs_idx
        Ra = body.center[senders]
        Rb = body.center[receivers]
        dR = - space.map_bond(displacement)(Ra, Rb)  # Looks like it should be opposite, but don't understand why
        prox = sensor(dR, body.orientation[senders], state.agent_state.proxs_dist_max[senders],
                      state.agent_state.proxs_cos_min[senders], len(state.agent_state.nve_idx), senders, mask)
        return state.agent_state.set(prox=prox)

    def sensorimotor(agent_state):
        motor = multi_switch(agent_state.behavior, agent_state.prox, agent_state.motor)
        return agent_state.set(motor=motor)

    def step_fn(state, neighbor, agent_neighs_idx):
        exists_mask = (state.nve_state.exists == 1)  # Only existing entities have effect on others
        state = state.set(agent_state=compute_prox(state, agent_neighs_idx, target_exists_mask=exists_mask))
        state = state.set(agent_state=sensorimotor(state.agent_state))
        force = force_fn(state, neighbor, exists_mask)
        state = physics_fn(state, force, shift, state.simulator_state.dt[0], neighbor=neighbor)
        return state

    return init_fn, step_fn