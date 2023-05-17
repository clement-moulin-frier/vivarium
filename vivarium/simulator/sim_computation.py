import numpy as np
import jax.numpy as jnp
from jax import ops, vmap, jit
import jax
from jax_md import space, energy, rigid_body, util, simulate

from collections import namedtuple
from jax_md.dataclasses import dataclass
from functools import partial
f32 = util.f32


Population = namedtuple('Population', ['position', 'theta', 'prox', 'motor', 'entity_type'])

@dataclass
class RigidRobot:
    center: util.Array
    orientation: util.Array
    prox: util.Array
    motor: util.Array
    entity_type: int
    def __getitem__(self, idx):
        return RigidRobot(self.center[idx], self.orientation[idx], self.prox[idx], self.motor[idx], self.entity_type)
    def to_rigid_body(self):
        return rigid_body.RigidBody(center=self.center, orientation=self.orientation)

@dataclass
class NVEState(simulate.NVEState):
    prox: util.Array
    motor: util.Array
    behavior: util.Array
    wheel_diameter: util.Array
    base_length: util.Array
    speed_mul: util.Array
    theta_mul: util.Array
    proxs_dist_max: util.Array
    proxs_cos_min: util.Array
    entity_type: util.Array

#
# util.register_custom_simulation_type(RigidRobot)
#
# #count_dof?
#
#
# @simulate.initialize_momenta.register(RigidRobot)
# def _(state, key, kT):
#     return simulate.initialize_momenta._registry[rigid_body.RigidBody](state, key, kT)
#
#
# @simulate.position_step.register(RigidRobot)
# def _(state, shift_fn, dt, m_rot=1, **kwargs):
#     return simulate.position_step._registry[rigid_body.RigidBody](state, shift_fn, dt, m_rot, **kwargs)
#
#
# @simulate.stochastic_step.register(RigidRobot)
# def _(state, dt: float, kT: float, gamma: float):
#     return simulate.stochastic_step._registry[rigid_body.RigidBody](state, dt, kT, gamma)
#
#
# @simulate.canonicalize_mass.register(RigidRobot)
# def _(state):
#     return simulate.canonicalize_mass._registry[rigid_body.RigidBody](state)
#
#
# @simulate.kinetic_energy.register(RigidRobot)
# def _(state) -> util.Array:
#     return simulate.kinetic_energy._registry[rigid_body.RigidBody](state)
#
#
# @simulate.temperature.register(RigidRobot)
# def _(state) -> util.Array:
#     return simulate.temperature._registry[rigid_body.RigidBody](state)


def get_verlet_force_fn(displacement, map_dim):
    # simulation_config = engine_config.simulation_config
    # agent_config = simulation_config.agent_configs[0]
    # displacement = engine_config.displacement
    # shift = engine_config.shift
    # map_dim = simulation_config.map_dim
    # n_agents = simulation_config.n_agents
    # dt = f32(simulation_config.dt)
    # speed_mul = agent_config.speed_mul
    # theta_mul = agent_config.theta_mul
    # proxs_dist_max = f32(agent_config.proxs_dist_max)
    # proxs_cos_min = f32(agent_config.proxs_cos_min)
    # base_length = f32(agent_config.base_length)
    # wheel_diameter = f32(agent_config.wheel_diameter)
    # entity_behaviors = engine_config.entity_behaviors
    # behavior_bank = engine_config.behavior_bank
    coll_force_fn = quantity.force(partial(total_collision_energy, displacement=displacement,
                                           epsilon=10., alpha=12))


    def collision_force(state, neighbor):
        return coll_force_fn(state.position.center, neighbor=neighbor, base_length=state.base_length)

    def friction_force(state, neighbor):
        cur_vel = state.momentum.center / state.mass.center
        return - f32(1e-1) * cur_vel

    def motor_force(state, neighbor):
        body = state.position
        fwd, rot = motor_command(state.motor, state.base_length, state.wheel_diameter)
        n = normal(body.orientation)
        cur_vel = state.momentum.center / state.mass.center
        cur_fwd_vel = vmap(jnp.dot)(cur_vel, n)
        cur_rot_vel = state.momentum.orientation / state.mass.orientation
        fwd_delta = fwd - cur_fwd_vel
        rot_delta = rot - cur_rot_vel
        fwd_force = f32(1e-1) * n * jnp.tile(fwd_delta, (map_dim, 1)).T
        rot_force = f32(1e-2) * rot_delta
        return rigid_body.RigidBody(center=fwd_force, orientation=rot_force)

    # motor = jnp.zeros((simulation_config.n_agents, 2), dtype=f32)  # Should be from manual input normally
    def force_fn(state, neighbor):
        # body = state.position
        # senders, receivers = neighbor.idx
        # Ra = body.center[senders]
        # Rb = body.center[receivers]
        # dR = - space.map_bond(displacement)(Ra, Rb) # Looks like it should be opposite, but don't understand why
        # proxs = sensor(dR, body.orientation[senders], proxs_dist_max, proxs_cos_min, neighbor)
        # motors = multi_switch(entity_behaviors, behavior_bank, proxs, state.motor)
        # fwd, rot = motor_command(state.motor, state.base_length, state.wheel_diameter)
        # n = normal(body.orientation)
        # cur_vel = state.momentum.center / state.mass.center
        # print(jnp.max(jnp.linalg.norm(cur_vel)))
        # if jnp.max(jnp.linalg.norm(cur_vel)) > 10.:
        #     print('high speed')
        # cur_fwd_vel = vmap(jnp.dot)(cur_vel, n)
        # cur_rot_vel = state.momentum.orientation / state.mass.orientation
        # fwd_delta = fwd - cur_fwd_vel
        # rot_delta = rot - cur_rot_vel
        # motor_fwd_force = n * jnp.tile(jnp.where(fwd_delta >= 0, 1., -1.), (map_dim, 1)).T
        # motor_fwd_force = f32(1e-1) * n * jnp.tile(fwd_delta, (map_dim, 1)).T
        # # fricton_fwd_force = - f32(1e-1) * cur_vel
        # fwd_force = collision_force(state, neighbor) + motor_fwd_force + fricton_fwd_force
        # rot_force = f32(1e-2) * rot_delta
        # if rot_force[0] > 4 or rot_force[0] < - 4:
        #     print('rot_force')
        mf = motor_force(state, neighbor)
        return rigid_body.RigidBody(center=collision_force(state, neighbor) + friction_force(state, neighbor) + mf.center,
                                    orientation=mf.orientation)

    return force_fn



# util.register_custom_simulation_type(RigidRobot)
def dynamics_rigid(displacement, shift, map_dim, dt, behavior_bank, force_fn=None, **sim_kwargs):
    force_fn = force_fn or get_verlet_force_fn(displacement, map_dim)
    # simulation_config = engine_config.simulation_config
    # agent_config = simulation_config.agent_configs[0]
    # box_size = simulation_config.box_size
    # shift = engine_config.shift
    # dt = engine_config.simulation_config.dt
    # map_dim = engine_config.simulation_config.map_dim
    # proxs_dist_max = agent_config.proxs_dist_max
    # proxs_cos_min = agent_config.proxs_cos_min
    # entity_behaviors = engine_config.entity_behaviors
    # behavior_bank = engine_config.behavior_bank
    shape = rigid_body.monomer
    def init_fn(key, positions, orientations, agent_configs_as_array_dict, kT=0.):
        key, subkey = jax.random.split(key)
        n_agents = positions.shape[0]
        print(n_agents)
        proxs = jnp.zeros((n_agents, 2))
        motors = jnp.zeros((n_agents, 2))
        body_positions = rigid_body.RigidBody(center=positions, orientation=orientations)
        force = rigid_body.RigidBody(center=jnp.zeros((n_agents, map_dim)), orientation=jnp.zeros(n_agents))
        state = simulate.NVEState(position=body_positions, momentum=None, force=force, mass=shape.mass())
        state = simulate.canonicalize_mass(state)
        state = simulate.initialize_momenta(state, key, kT)
        return NVEState(position=state.position, momentum=state.momentum, force=state.force, mass=state.mass,
                        prox=proxs, motor=motors, **agent_configs_as_array_dict)

    def physics_fn(state, force, shift_fn, dt, neighbor):
        """Apply a single step of velocity Verlet integration to a state."""
        dt = f32(dt)
        dt_2 = f32(dt / 2)

        # state = sensorimotor(state, neighbor)  # now in step_fn

        state = simulate.momentum_step(state, dt_2)
        state = simulate.position_step(state, shift_fn, dt, neighbor=neighbor)
        state = state.set(force=force)
        state = simulate.momentum_step(state, dt_2)

        return state

    def sensorimotor(state, neighbor):
        body = state.position
        senders, receivers = neighbor.idx
        Ra = body.center[senders]
        Rb = body.center[receivers]
        dR = - space.map_bond(displacement)(Ra, Rb) # Looks like it should be opposite, but don't understand why
        prox = sensor(dR, body.orientation[senders], state.proxs_dist_max[senders], state.proxs_cos_min[senders], neighbor)
        # print(state.behavior, behavior_bank, prox, state.motor)
        motor = multi_switch(state.behavior, behavior_bank, prox, state.motor)

        return state.set(prox=prox, motor=motor)
    def step_fn(state, neighbor, **kwargs):
        # _dt = kwargs.pop('dt', dt)
        state = sensorimotor(state, neighbor)
        force = force_fn(state, neighbor)
        return physics_fn(state, force, shift, dt, neighbor=neighbor)

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


def sensor(displ, theta, dist_max, cos_min, neighbors):
    proxs = ops.segment_max(sensor_fn(displ, theta, dist_max, cos_min),
                            neighbors.idx[0], len(neighbors.reference_position))
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

multi_switch = jax.vmap(jax.lax.switch, (0, None, 0, 0))

import jax.numpy as jnp
from jax_md import energy, quantity
from jax import lax

# strength = 0.01

# def collision_energy(displ, base_length, epsilon, alpha, **kwargs):
#   # distance = jnp.linalg.norm(ag1 - ag2)
#   return energy.soft_sphere(jnp.linalg.norm(displ), sigma=base_length * f32(2.), epsilon=epsilon, alpha=alpha)
#
# collision_energy = vmap(collision_energy, (0, 0, None, None))
#
def collision_energy(displacement_fn, r_a, r_b, l_a, l_b, epsilon, alpha):
    dist = jnp.linalg.norm(displacement_fn(r_a, r_b))
    sigma = l_a + l_b
    return energy.soft_sphere(dist, sigma=sigma, epsilon=epsilon, alpha=alpha)

collision_energy = vmap(collision_energy, (None, 0, 0, 0, 0, None, None))

def total_collision_energy(positions, base_length, neighbor, displacement, epsilon=1e-2, alpha=2, **kwargs):
    lax.stop_gradient(base_length)
    senders, receivers = neighbor.idx
    Ra = positions[senders]
    Rb = positions[receivers]
    l_a = base_length[senders]
    l_b = base_length[receivers]
    e = collision_energy(displacement, Ra, Rb, l_a, l_b, epsilon, alpha)
    # dR = -space.map_bond(displacement)(Ra, Rb)
    # e = collision_energy(dR, base_length, epsilon, alpha)
    #e = ops.segment_sum(e, neighbors.idx[0], len(neighbors.reference_position))
    # print('e.shape', e.shape)
    return jnp.sum(e)

def dynamics(simulation_config, agent_config):

    displacement = simulation_config.displacement
    shift = simulation_config.shift
    map_dim = simulation_config.map_dim
    dt = simulation_config.dt

    speed_mul = agent_config.speed_mul
    theta_mul = agent_config.theta_mul
    proxs_dist_max = agent_config.proxs_dist_max
    proxs_cos_min = agent_config.proxs_cos_min
    base_length = agent_config.base_length
    wheel_diameter = agent_config.wheel_diameter

    entity_behaviors = simulation_config.entity_behaviors()
    behavior_bank = simulation_config.behavior_bank

    def move(positions, thetas, fwd, rot, dpos):
        n = normal(thetas)
        return (shift(positions, dt * (dpos + speed_mul * n * jnp.tile(fwd, (map_dim, 1)).T)),
                thetas + dt * rot * theta_mul)

    def update(_, state_neighbors):

        print("update")
        state, neighs = state_neighbors

        neighbors = neighs.update(state.position)

        senders, receivers = neighbors.idx
        Ra = state.position[senders]
        Rb = state.position[receivers]

        dR = - space.map_bond(displacement)(Ra, Rb) # Looks like it should be opposite, but don't understand why

        proxs = sensor(dR, state.theta[senders], proxs_dist_max, proxs_cos_min, neighbors)

        motors = multi_switch(entity_behaviors, behavior_bank, proxs, state.motor)

        fwd, rot = motor_command(motors, base_length, wheel_diameter)

        dpos = quantity.force(total_collision_energy)(state.position, base_length, neighbors, displacement)

        new_entity_state = Population(*move(state.position, state.theta, fwd, rot, dpos=dpos), prox=proxs, motor=motors,
                                      entity_type=state.entity_type)

        return new_entity_state, neighs

    return update
