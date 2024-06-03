import logging as lg
from enum import Enum
from functools import partial
from typing import Tuple

import jax.numpy as jnp

from jax import  jit
from flax import struct
from jax_md import simulate


# TODO : The best is surely to only define BaseState because some envs might not use EntityState / ObjectState or AgentState
class EntityType(Enum):
    AGENT = 0
    OBJECT = 1

# No need to define position, momentum, force, and mass (i.e already in jax_md.simulate.NVEState)
@struct.dataclass
class BaseEntityState(simulate.NVEState):
    entity_type: jnp.array
    entity_idx: jnp.array
    diameter: jnp.array
    friction: jnp.array
    exists: jnp.array

    @property
    def velocity(self) -> jnp.array:
        return self.momentum / self.mass
    
@struct.dataclass
class BaseAgentState:
    ent_idx: jnp.array
    color: jnp.array

@struct.dataclass
class BaseObjectState:
    ent_idx: jnp.array 
    color: jnp.array

@struct.dataclass
class BaseState:
    time: jnp.int32
    box_size: jnp.int32
    max_agents: jnp.int32
    max_objects: jnp.int32
    neighbor_radius: jnp.float32
    dt: jnp.float32  # Give a more explicit name
    collision_alpha: jnp.float32
    collision_eps: jnp.float32
    entities: BaseEntityState
    agents: BaseAgentState
    objects: BaseObjectState


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
    