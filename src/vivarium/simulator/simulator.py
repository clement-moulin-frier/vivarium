# from jax import jit, vmap
# import jax.numpy as jnp
# import numpy as np
# from collections import namedtuple
# from tranquilizer import tranquilize
#
# global x
#
# n = 10
#
# x = jnp.zeros(10)
#
# def compute(i):
#     j = i + 1
#
#     return jnp.array(j)
#
# compute = vmap(compute)
#
# compute = jit(compute)
#
# @tranquilize()
# def publish():
#     global x
#
#     y = compute(x)
#
#     #x = x.at[0].set(y + 1)
#
#     return {'arr': np.array(y).tolist()}

import numpy as np

# from jax.config import config ; config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import random
from jax import jit
from jax import vmap
from jax import lax
from jax import ops

#vectorize = np.vectorize

from functools import partial
from collections import namedtuple
#import base64

import os

from jax_md import space, smap, energy, minimize, quantity, simulate, partition, util
# from jax_md.util import f32


# normalize = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)

from enum import Enum
from dataclasses import dataclass

from tranquilizer import tranquilize

box_size = 100.0
map_dim = 2
wheel_diameter = 2.
base_lenght = 10.



Population = namedtuple('Population', ['positions', 'thetas', 'entity_type'])

PopulationObstacle = namedtuple('Population', ['positions', 'thetas', 'entity_type', 'diameters'])

EntityType = Enum('EntityType', ['PREY', 'PREDATOR', 'OBSTACLE'])

key = random.PRNGKey(0)


speed_mul = 0.1
theta_mul = 0.1

pop_config = {EntityType.PREY: 20, EntityType.PREDATOR: 2, EntityType.OBSTACLE: 2}


def generate_entities(dict_type_to_count, box_size = box_size, map_dim=map_dim, key=key):
  key, subkey = random.split(key)
  pops = {}
  for entity_type, count in dict_type_to_count.items():
    key, subkey = random.split(key)
    positions = box_size * random.uniform(subkey, (count, map_dim))
    key, subkey = random.split(key)
    thetas = random.uniform(subkey, (count,), maxval=2 * np.pi)
    pops[entity_type.name] = Population(positions, thetas, jnp.array(entity_type.value))
    if entity_type == EntityType.OBSTACLE:
        diameters = 10. * jnp.ones((count,))
        pops[entity_type.name] = PopulationObstacle(positions, thetas, jnp.array(entity_type.value), diameters)
    else:
        pops[entity_type.name] = Population(positions, thetas, jnp.array(entity_type.value))
  return pops, key

populations, key = generate_entities(pop_config)

@tranquilize()
def get_sim_config():

    sim_config = dict(
        box_size=box_size,
        map_dim=map_dim,
        wheel_diameter=wheel_diameter,
        base_lenght=base_lenght,
        pop_config={e_type.name: n for e_type, n in pop_config.items()}
    )
    return sim_config

num_steps_lax = 50
num_lax_loops = 1

displacement, shift = space.periodic(box_size)
neighbor_radius = box_size

neighbor_fn = partition.neighbor_list(displacement,
                                      box_size,
                                      r_cutoff=neighbor_radius,
                                      dr_threshold=10.,
                                      capacity_multiplier=1.5,
                                      format=partition.Sparse)

global neighbors
neighbors = neighbor_fn.allocate(populations[EntityType.PREY.name].positions)


def sensor_fn(dist_max, cos_min):
  def f(displ, theta):
    print("f", displ.shape, theta)
    dist = jnp.linalg.norm(displ)
    n = jnp.array([jnp.cos( - theta), jnp.sin(- theta)])
    rot_matrix = jnp.array([[n[0], - n[1]], [n[1], n[0]]])
    rot_displ = jnp.dot(rot_matrix, jnp.reshape(displ, (2, 1))).reshape((-1, ))
    cos_dir = rot_displ[0] / dist
    prox = 1. - (dist / dist_max)
    #print('sensor', dist, cos_dir, prox)
    in_view = jnp.logical_and(dist < dist_max, cos_dir > cos_min)
    at_left = jnp.logical_and(True, rot_displ[1] >= 0)
    left = in_view * at_left * prox
    right = in_view * (1. - at_left) * prox
    return jnp.array([left, right])

  return f

f = sensor_fn(box_size, 0.)
f = vmap(f)

def sensor(displ, theta, neighbors):
  proxs = ops.segment_max(f(displ, theta), neighbors.idx[0], len(neighbors.reference_position))
  return proxs



def lr_2_fwd_rot(left_spd, right_spd):
    fwd = (wheel_diameter / 4.) * (left_spd + right_spd)
    rot = 0.5 * (wheel_diameter / base_lenght) * (right_spd - left_spd)
    return fwd, rot

def fwd_rot_2_lr(fwd, rot):
    left = ((2.0 * fwd) - (rot * base_lenght)) / (wheel_diameter)
    right = ((2.0 * fwd) + (rot * base_lenght)) / (wheel_diameter)
    return left, right

def motor_command(wheel_activation):
  fwd, rot = lr_2_fwd_rot(wheel_activation[0], wheel_activation[1])
  print('motors', fwd, rot)
  return fwd, rot


def normal(theta):
  return jnp.array([jnp.cos(theta), jnp.sin(theta)])

normal = vmap(normal)

def cross(array):
  return jnp.hstack((array[:, -1:], array[:, :1]))

def dynamics(dt):
  def move(boids, fwd, rot):
    R, theta, *_ = boids
    n = normal(theta)
    return (shift(R, dt * speed_mul * n * jnp.tile(fwd, (map_dim, 1)).T),
            theta + dt * rot * theta_mul)

  @jit
  def update(_, state_and_neighbors):

    state, neighbors = state_and_neighbors
    boids = state[EntityType.PREY.name]

    neighbors = neighbors.update(boids.positions)

    senders, receivers = neighbors.idx
    Ra = boids.positions[senders]
    Rb = boids.positions[receivers]

    dR = - space.map_bond(displacement)(Ra, Rb) # Looks like it should be opposite, but don't understand why

    proxs = sensor(dR, boids.thetas[senders], neighbors)

    motors = cross(proxs) # Braitenberg simple

    fwd, rot = vmap(motor_command)(motors)

    state[EntityType.PREY.name] = Population(*move(boids, fwd, rot), jnp.array(EntityType.PREY.value))

    return state, neighbors

  return update

update = dynamics(dt=1e-1)

global state
state = populations

@tranquilize()
def test():
    #x = state.shape
    num_lax_loops += 1
    return num_lax_loops

def serialize_state(s):
    serial_s = {}

    for type, pop in s.items():
        serial_pop_kwargs = {}
        for field, jarray in pop._asdict().items():
            serial_pop_kwargs[field] = np.array(jarray).tolist()
        # if isinstance(s[type], Population):
        #     serial_pop = Population(**serial_pop_kwargs)
        # elif isinstance(s[type], PopulationObstacle):
        #     serial_pop = PopulationObstacle(**serial_pop_kwargs)
        serial_s[type] = serial_pop_kwargs
    return serial_s

@tranquilize(method='post')
def get_state():
    global state
    print('state')
    return serialize_state(state)

global sim_start
sim_start = False

@tranquilize()
def start():
    global sim_start
    sim_start = True

@tranquilize()
def stop():
    global sim_start
    sim_start = False

@tranquilize()
def run():
    global state, neighbors, sim_start
    print('start run')
    boids_buffer = []

    while True:
        if not sim_start:
            continue

        new_state, neighbors = lax.fori_loop(0, num_lax_loops, update, (state, neighbors))

        # If the neighbor list can't fit in the allocation, rebuild it but bigger.
        if neighbors.did_buffer_overflow:
            print('REBUILDING')
            neighbors = neighbor_fn.allocate(state[EntityType.PREY.name].R)
            state, neighbors = lax.fori_loop(0, 50, update, (state, neighbors))
            assert not neighbors.did_buffer_overflow
        else:
            state = new_state

        boids_buffer += [state[EntityType.PREY.name]]
    print('stop run')
    return "test" #np.array(state).tolist()



# # import sys
# #
# # if __name__ == '__main__':
# #     sys.exit(run())