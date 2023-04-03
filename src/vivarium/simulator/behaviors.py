import jax.numpy as jnp
from enum import Enum
from jax import vmap

linear_behavior_enum = Enum('matrices', ['FEAR', 'AGGRESSION', 'LOVE', 'SHY'])

linear_behavior_matrices = {linear_behavior_enum.FEAR: jnp.array([[1., 0., 0.], [0., 1., 0.]]),
                            linear_behavior_enum.AGGRESSION: jnp.array([[0., 1., 0.], [1., 0., 0.]]),
                            linear_behavior_enum.LOVE: jnp.array([[-1., 0., 1.], [0., -1., 1.]]),
                            linear_behavior_enum.SHY: jnp.array([[0., -1., 1.], [-1., 0., 1.]]),
                            }


def linear_behavior(proxs, matrix):
    return matrix.dot(jnp.hstack((proxs, 1.)))


def _single_weighted_behavior(w, b):
    f = vmap(lambda w, b: w * b)
    return f(w, b).sum(axis=0)


def weighted_behavior(weights, behavior_set, proxs):
    return _single_weighted_behavior(weights, behavior_set).dot(jnp.hstack((proxs, 1.)))

#behavior = vmap(behavior, (0, None, 0))


def apply_motors(proxs, motors):
    return motors

def noop(proxs):
    return jnp.array([0., 0.])