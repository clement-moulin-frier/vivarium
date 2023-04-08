import jax.numpy as jnp
from jax import ops, vmap
import jax
from jax_md import space, energy

from collections import namedtuple


Population = namedtuple('Population', ['positions', 'thetas', 'proxs', 'motors', 'entity_type'])


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


sensor_fn = vmap(sensor_fn, (0, 0, None, None))


def sensor(displ, theta, dist_max, cos_min, neighbors):
    proxs = ops.segment_max(sensor_fn(displ, theta, dist_max, cos_min), neighbors.idx[0], len(neighbors.reference_position))
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


motor_command = vmap(motor_command, (0, None, None))


def normal(theta):
  return jnp.array([jnp.cos(theta), jnp.sin(theta)])


normal = vmap(normal)

multi_switch = jax.vmap(jax.lax.switch, (0, None, 0, 0))

import jax.numpy as jnp
from jax_md import energy, quantity
from jax import lax
strength = 1000.0

def collision_energy(displ, base_length, **kwargs):
  # distance = jnp.linalg.norm(ag1 - ag2)
  return energy.soft_sphere(jnp.linalg.norm(displ), sigma=base_length + 1.0, epsilon=strength)

collision_energy = vmap(collision_energy, (0, None))




def dynamics(engine_config, simulation_config, agent_config):

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

    entity_behaviors = jnp.array(simulation_config.entity_behaviors, dtype=int)
    behavior_bank = engine_config.behavior_bank
    def total_collision_energy(positions, base_length, neighbors, **kwargs):
        lax.stop_gradient(base_length)
        senders, receivers = neighbors.idx
        Ra = positions[senders]
        Rb = positions[receivers]
        dR = -space.map_bond(displacement)(Ra, Rb)
        e = collision_energy(dR, base_length)
        #e = ops.segment_sum(e, neighbors.idx[0], len(neighbors.reference_position))
        # print('e.shape', e.shape)
        return jnp.sum(e)
    def move(positions, thetas, fwd, rot, dpos):
        n = normal(thetas)
        return (shift(positions, dt * (dpos + speed_mul * n * jnp.tile(fwd, (map_dim, 1)).T)),
                thetas + dt * rot * theta_mul)

    def update(_, state_neighbors):

        print("update")
        state, neighs = state_neighbors

        neighbors = neighs.update(state.positions)

        senders, receivers = neighbors.idx
        Ra = state.positions[senders]
        Rb = state.positions[receivers]

        dR = - space.map_bond(displacement)(Ra, Rb) # Looks like it should be opposite, but don't understand why

        proxs = sensor(dR, state.thetas[senders], proxs_dist_max, proxs_cos_min, neighbors)

        motors = multi_switch(entity_behaviors, behavior_bank, proxs, state.motors)

        fwd, rot = motor_command(motors, base_length, wheel_diameter)

        dpos = quantity.force(total_collision_energy)(state.positions, base_length, neighbors)

        new_entity_state = Population(*move(state.positions, state.thetas, fwd, rot, dpos=dpos), proxs=proxs, motors=motors,
                                      entity_type=state.entity_type)

        return new_entity_state, neighs

    return update
