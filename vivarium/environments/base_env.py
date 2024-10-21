import logging as lg

from functools import partial
from typing import Tuple

import jax.numpy as jnp

from jax import  jit
from flax import struct


@struct.dataclass
class BaseState:
    time: jnp.int32
    box_size: jnp.int32
   

class BaseEnv:
    def __init__(self):
        raise(NotImplementedError)

    def init_state(self) -> BaseState:
        raise(NotImplementedError)

    @partial(jit, static_argnums=(0,))
    def _step(self, state: BaseState, neighbors: jnp.array) -> Tuple[BaseState, jnp.array]:
        raise(NotImplementedError)
    
    def step(self, state: BaseState) -> BaseState:
        current_state = state
        state, neighbors = self._step(current_state, self.neighbors)

        if self.neighbors.did_buffer_overflow:
            # reallocate neighbors and run the simulation from current_state
            lg.warning('BUFFER OVERFLOW: rebuilding neighbors')
            neighbors = self.allocate_neighbors(state)
            assert not neighbors.did_buffer_overflow

        self.neighbors = neighbors
        return state

    def allocate_neighbors(self, state, position=None):
        position = state.entities.position.center if position is None else position
        neighbors = self.neighbor_fn.allocate(position)
        return neighbors
    