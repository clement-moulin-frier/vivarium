import jax
from jax import ops, vmap, lax
import jax.numpy as jnp

from jax_md import space, rigid_body, util, simulate, energy, quantity
from jax_md.dataclasses import dataclass

from functools import partial
from enum import Enum
from itertools import compress


f32 = util.f32


class EntityType(Enum):
    AGENT = 0
    OBJECT = 1

@dataclass
class NVEState(simulate.NVEState):
    entity_type: util.Array
    entity_idx: util.Array  # idx in XState (e.g. AgentState)
    diameter: util.Array
    friction: util.Array

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
class State:
    nve_state: NVEState
    agent_state: AgentState
    object_state: ObjectState

    def field(self, etype):
        name = etype.name.lower()
        return getattr(self, f'{name}_state')

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
        return nve_idx if field == 'nve_state' else self.nve_state.entity_idx[nve_idx]  # self.field(etype).nve_idx[entity_idx]

    def __getattr__(self, name):
        def wrapper(e_type):
            value = getattr(self.nve_state, name)
            if isinstance(value, rigid_body.RigidBody):
                return rigid_body.RigidBody(center=value.center[self.e_cond(e_type)],
                                            orientation=value.orientation[self.e_cond(e_type)])
            else:
                return value[self.e_cond(e_type)]
        return wrapper


def switch_fn(fn_list):
    def switch(index, *operands):
        return jax.lax.switch(index, fn_list, *operands)
    return switch



def get_verlet_force_fn(displacement, map_dim):
    coll_force_fn = quantity.force(partial(total_collision_energy, displacement=displacement,
                                           epsilon=10., alpha=12))

    def collision_force(nve_state, neighbor):
        return coll_force_fn(nve_state.position.center, neighbor=neighbor, diameter=nve_state.diameter)

    def friction_force(nve_state):
        cur_vel = nve_state.momentum.center / nve_state.mass.center
        return - jnp.tile(nve_state.friction, (map_dim, 1)).T * cur_vel

    def motor_force(state):
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
        fwd_force = n * jnp.tile(fwd_delta, (map_dim, 1)).T * jnp.tile(state.agent_state.speed_mul, (map_dim, 1)).T
        rot_force = rot_delta * state.agent_state.theta_mul

        return rigid_body.RigidBody(center=jnp.zeros_like(state.nve_state.position.center).at[agent_idx].set(fwd_force),
                                    orientation=jnp.zeros_like(state.nve_state.position.orientation).at[agent_idx].set(rot_force))

    def force_fn(state, neighbor):
        mf = motor_force(state)
        return rigid_body.RigidBody(center=collision_force(state.nve_state, neighbor) + friction_force(state.nve_state) + mf.center,
                                    orientation=mf.orientation)

    return force_fn


def dynamics_rigid(displacement, shift, map_dim, dt, behavior_bank, force_fn=None, **sim_kwargs):
    force_fn = force_fn or get_verlet_force_fn(displacement, map_dim)
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
        dt = f32(dt)
        dt_2 = f32(dt / 2)
        # state = sensorimotor(state, neighbor)  # now in step_fn
        nve_state = simulate.momentum_step(state.nve_state, dt_2)
        nve_state = simulate.position_step(nve_state, shift_fn, dt, neighbor=neighbor)
        nve_state = nve_state.set(force=force)
        nve_state = simulate.momentum_step(nve_state, dt_2)

        return state.set(nve_state=nve_state)

    def compute_prox(state, agent_neighs_idx):
        body = state.nve_state.position
        senders, receivers = agent_neighs_idx
        Ra = body.center[senders]
        Rb = body.center[receivers]
        dR = - space.map_bond(displacement)(Ra, Rb) # Looks like it should be opposite, but don't understand why
        prox = sensor(dR, body.orientation[senders], state.agent_state.proxs_dist_max[senders],
                      state.agent_state.proxs_cos_min[senders], len(state.agent_state.nve_idx), senders)
        return state.agent_state.set(prox=prox)

    def sensorimotor(agent_state):

        motor = multi_switch(agent_state.behavior, agent_state.prox, agent_state.motor)

        return agent_state.set(motor=motor)

    def step_fn(state, neighbor, agent_neighs_idx, **kwargs):
        state = state.set(agent_state=compute_prox(state, agent_neighs_idx))
        state = state.set(agent_state=sensorimotor(state.agent_state))
        force = force_fn(state, neighbor)
        state = physics_fn(state, force, shift, dt, neighbor=neighbor)
        return state

    return init_fn, step_fn


def sensor_fn(displ, theta, dist_max, cos_min):
    dist = jnp.linalg.norm(displ)
    n = jnp.array([jnp.cos( - theta), jnp.sin(- theta)])
    rot_matrix = jnp.array([[n[0], - n[1]], [n[1], n[0]]])
    rot_displ = jnp.dot(rot_matrix, jnp.reshape(displ, (2, 1))).reshape((-1, ))
    cos_dir = rot_displ[0] / dist
    prox = 1. - (dist / dist_max)
    in_view = jnp.logical_and(dist < dist_max, cos_dir > cos_min)
    at_left = jnp.logical_and(True, rot_displ[1] >= 0)
    left = in_view * at_left * prox
    right = in_view * (1. - at_left) * prox
    return jnp.array([left, right])


sensor_fn = vmap(sensor_fn, (0, 0, 0, 0))


def sensor(displ, theta, dist_max, cos_min, n_agents, senders):
    proxs = ops.segment_max(sensor_fn(displ, theta, dist_max, cos_min),
                            senders, n_agents)
    return proxs


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


def normal(theta):
  return jnp.array([jnp.cos(theta), jnp.sin(theta)])


normal = vmap(normal)

def collision_energy(displacement_fn, r_a, r_b, l_a, l_b, epsilon, alpha):
    dist = jnp.linalg.norm(displacement_fn(r_a, r_b))
    sigma = l_a + l_b
    return energy.soft_sphere(dist, sigma=sigma, epsilon=epsilon, alpha=alpha)


collision_energy = vmap(collision_energy, (None, 0, 0, 0, 0, None, None))


def total_collision_energy(positions, diameter, neighbor, displacement, epsilon=1e-2, alpha=2, **kwargs):
    diameter = lax.stop_gradient(diameter)
    senders, receivers = neighbor.idx
    Ra = positions[senders]
    Rb = positions[receivers]
    l_a = diameter[senders]
    l_b = diameter[receivers]
    e = collision_energy(displacement, Ra, Rb, l_a, l_b, epsilon, alpha)
    return jnp.sum(e)

