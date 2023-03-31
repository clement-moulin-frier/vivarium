import jax.numpy as jnp
from jax import ops, vmap, jit

from jax_md import space

from collections import namedtuple


Population = namedtuple('Population', ['positions', 'thetas', 'entity_type'])

PopulationObstacle = namedtuple('Population', ['positions', 'thetas', 'entity_type', 'diameters'])



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

from jax import debug

def sensor(displ, theta, dist_max, cos_min, neighbors):
    proxs = ops.segment_max(sensor_fn(displ, theta, dist_max, cos_min), neighbors.idx[0], len(neighbors.reference_position))
    return proxs



def lr_2_fwd_rot(left_spd, right_spd, base_length, wheel_diameter):
    fwd = (wheel_diameter / 4.) * (left_spd + right_spd)
    rot = 0.5 * (wheel_diameter / base_length) * (right_spd - left_spd)
    return fwd, rot

def fwd_rot_2_lr(fwd, rot, base_length, wheel_diameter):
    left = ((2.0 * fwd) - (rot * base_length)) / (wheel_diameter)
    right = ((2.0 * fwd) + (rot * base_length)) / (wheel_diameter)
    return left, right

def motor_command(wheel_activation, base_length, wheel_diameter):
  fwd, rot = lr_2_fwd_rot(wheel_activation[0], wheel_activation[1], base_length, wheel_diameter)
  return fwd, rot

motor_command = vmap(motor_command, (0, None, None))

def normal(theta):
  return jnp.array([jnp.cos(theta), jnp.sin(theta)])

normal = vmap(normal)


def cross(array):
  return jnp.hstack((array[:, -1:], array[:, :1]))

def noop(proxs):
    return jnp.zeros(proxs.shape[0]), jnp.zeros(proxs.shape[0])
def fear(proxs):
    motors = proxs # Braitenberg simple
    #fwd, rot = motor_command(motors)
    return motors

def agression(proxs):
    motors = cross(proxs) # Braitenberg simple
    #fwd, rot = motor_command(motors)
    return motors


def weighted_behavior(w, b):
    f = vmap(lambda w, b: w * b)
    return f(w, b).sum(axis=0)


def behavior(weights, behavior_set, proxs):
    return weighted_behavior(weights, behavior_set).dot(jnp.hstack((proxs, 1.)))

behavior = vmap(behavior, (0, None, 0))

def dynamics(sim_config, pop_config, beh_config, entity_slices, shift, displacement, map_dim, base_length, wheel_diameter, proxs_dist_max, proxs_cos_min, speed_mul=1., theta_mul=1., dt=1e-1):

    behavior_set, behavior_map = beh_config
    def move(positions, thetas, fwd, rot):
        n = normal(thetas)
        return (shift(positions, dt * speed_mul * n * jnp.tile(fwd, (map_dim, 1)).T),
                thetas + dt * rot * theta_mul)

    def update(_, state_neighbors):

        state, neighs = state_neighbors

        #all_positions, all_thetas, all_etypes = state

        neighbors = neighs.update(state.positions)

        senders, receivers = neighbors.idx
        Ra = state.positions[senders]
        Rb = state.positions[receivers]

        dR = - space.map_bond(displacement)(Ra, Rb) # Looks like it should be opposite, but don't understand why

        proxs = sensor(dR, state.thetas[senders], proxs_dist_max, proxs_cos_min, neighbors)


        motors = behavior(behavior_map, behavior_set, proxs)

        # motors = jnp.zeros((all_positions.shape[0], 2))
        #
        # for etype, behavior in beh_config.items():
        #     motors[slice(*entity_slices[etype]), :] = behavior(proxs[slice(*entity_slices[etype]), :] #See if can be done as a matrix operation, e.g. using https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tensordot.html

        fwd, rot = motor_command(motors, base_length, wheel_diameter)
            # fwd, rot = behavior(proxs)

        new_entity_state = Population(*move(state.positions, state.thetas, fwd, rot), state.entity_type)

        return new_entity_state, neighs

    return update

