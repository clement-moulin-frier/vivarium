import jax.numpy as jnp
from enum import Enum
from jax import vmap
from functools import partial

linear_behavior_enum = Enum('matrices', ['FEAR', 'AGGRESSION', 'LOVE', 'SHY'])

linear_behavior_matrices = {linear_behavior_enum.FEAR: jnp.array([[1., 0., 0.], [0., 1., 0.]]),
                            linear_behavior_enum.AGGRESSION: jnp.array([[0., 1., 0.], [1., 0., 0.]]),
                            linear_behavior_enum.LOVE: jnp.array([[-1., 0., 1.], [0., -1., 1.]]),
                            linear_behavior_enum.SHY: jnp.array([[0., -1., 1.], [-1., 0., 1.]]),
                            }


def linear_behavior(proxs, motors, matrix):
    return matrix.dot(jnp.hstack((proxs, 1.)))


def apply_motors(proxs, motors):
    return motors


def noop(proxs, motors):
    return jnp.array([0., 0.])


behavior_bank = [partial(linear_behavior, matrix=linear_behavior_matrices[beh])
                 for beh in linear_behavior_enum] \
                + [apply_motors, noop]

behavior_name_map = {beh.name: i for i, beh in enumerate(linear_behavior_enum)}
behavior_name_map['manual'] = len(behavior_bank) - 2
behavior_name_map['noop'] = len(behavior_bank) - 1
