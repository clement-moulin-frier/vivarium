from enum import Enum

import jax.numpy as jnp

from jax_md.dataclasses import dataclass as md_dataclass
from jax_md import simulate

from vivarium.environments.base_env import BaseState, BaseEnv


class EntityType(Enum):
    AGENT = 0
    OBJECT = 1

# Already incorporates position, momentum, force, mass and velocity
@md_dataclass
class EntityState(simulate.NVEState):
    entity_type: jnp.array
    entity_idx: jnp.array
    diameter: jnp.array
    friction: jnp.array
    exists: jnp.array
    
@md_dataclass
class ParticleState:
    ent_idx: jnp.array
    color: jnp.array

@md_dataclass
class AgentState(ParticleState):
    prox: jnp.array
    motor: jnp.array
    proximity_map_dist: jnp.array
    proximity_map_theta: jnp.array
    behavior: jnp.array
    params: jnp.array
    wheel_diameter: jnp.array
    speed_mul: jnp.array
    max_speed: jnp.array
    theta_mul: jnp.array    
    proxs_dist_max: jnp.array
    proxs_cos_min: jnp.array

@md_dataclass
class ObjectState(ParticleState):
    pass

@md_dataclass
class State(BaseState):
    max_agents: jnp.int32
    max_objects: jnp.int32
    neighbor_radius: jnp.float32
    dt: jnp.float32  # Give a more explicit name
    collision_alpha: jnp.float32
    collision_eps: jnp.float32
    entities: EntityState
    agents: AgentState
    objects: ObjectState    

class Behaviors(Enum):
    FEAR = 0
    AGGRESSION = 1
    LOVE = 2
    SHY = 3
    NOOP = 4
    MANUAL = 5

behavior_params = {
    Behaviors.FEAR.value: jnp.array(
        [[1., 0., 0.], 
         [0., 1., 0.]]),
    Behaviors.AGGRESSION.value: jnp.array(
        [[0., 1., 0.], 
         [1., 0., 0.]]),
    Behaviors.LOVE.value: jnp.array(
        [[-1., 0., 1.], 
         [0., -1., 1.]]),
    Behaviors.SHY.value: jnp.array(
        [[0., -1., 1.], 
         [-1., 0., 1.]]),
    Behaviors.NOOP.value: jnp.array(
        [[0., 0., 0.], 
         [0., 0., 0.]]),
    Behaviors.MANUAL.value: jnp.array(
        [[0., 0., 0.], 
         [0., 0., 0.]])
}

def behavior_to_params(behavior):
    """Return the params associated to a behavior.

    :param behavior: behavior id (int)
    :return: params
    """
    return behavior_params[behavior]