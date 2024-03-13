from enum import Enum

import matplotlib.colors as mcolors
import jax.numpy as jnp 

from jax import random
from jax_md import util, simulate, rigid_body
from jax_md.dataclasses import dataclass
from jax_md.rigid_body import RigidBody


# Helper function to transform a color string into rgb with matplotlib colors
def string_to_rgb(color_str):
    return jnp.array(list(mcolors.to_rgb(color_str)))

# TODO : Add documentation on these classes 

class EntityType(Enum):
    AGENT = 0
    OBJECT = 1

    def to_state_type(self):
        return StateType(self.value)

class StateType(Enum):
    AGENT = 0
    OBJECT = 1
    SIMULATOR = 2

    def is_entity(self):
        return self != StateType.SIMULATOR

    def to_entity_type(self):
        assert self.is_entity()
        return EntityType(self.value)
    

# NVE (should maybe rename it entities) 

# No need to define position, momentum, force, and mass (i.e already in simulate.NVEState)
@dataclass
class NVEState(simulate.NVEState):
    entity_type: util.Array
    entity_idx: util.Array  # idx in XState (e.g. AgentState)
    diameter: util.Array
    friction: util.Array
    exists: util.Array

    @property
    def velocity(self) -> util.Array:
        return self.momentum / self.mass
    
def init_nve_state(
        simulator_state,
        diameter,
        friction,
        seed,
        ) -> NVEState:
    """
    Initialize agent state with given parameters
    """
    n_agents = simulator_state.n_agents[0]
    n_objects = simulator_state.n_objects[0]
    n_entities = n_agents + n_objects

    key = random.PRNGKey(seed)
    key_pos, key_or = random.split(key, 2)

    # Assign random positions to each entities (will be changed in the future to allow defining custom positions)
    positions = random.uniform(key_pos, (n_entities, 2)) * simulator_state.box_size
    # Assign random orientations between 0 and 2*pi
    orientations = random.uniform(key_or, (n_entities, 2)) * 2 * jnp.pi

    return NVEState(
        position=RigidBody(center=positions, orientation=orientations),
        # TODO: Why is momentum set to none ? 
        momentum=None,
        # Should we indeed set the force and mass to 0 ?
        force=RigidBody(center=jnp.zeros((n_entities, 2)), orientation=jnp.zeros(n_entities)),
        mass=RigidBody(center=jnp.zeros((n_entities, 1)), orientation=jnp.zeros(n_entities)),
        entity_type=jnp.array([EntityType.AGENT.value] * n_agents + [EntityType.OBJECT.value] * n_objects, dtype=int),
        entity_idx = jnp.array(list(range(n_agents)) + list(range(n_objects))),
        diameter=jnp.full((n_entities), diameter),
        friction=jnp.full((n_entities), friction),
        # Set all the entities to exist by default, but we should add a function to be able to change that 
        exists=jnp.ones(n_entities, dtype=int)
        )


# Agents 

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

# Could implement it as a static or class method 
def init_agent_state(
        n_agents,
        behavior,
        wheel_diameter,
        speed_mul,
        theta_mul,
        prox_dist_max,
        prox_cos_min,
        color
        ) -> AgentState:
    """
    Initialize agent state with given parameters
    """
    return AgentState(
        nve_idx=jnp.arange(n_agents, dtype=int),
        prox=jnp.zeros((n_agents, 2)),
        motor=jnp.zeros((n_agents, 2)),
        behavior=jnp.full((n_agents), behavior),
        wheel_diameter=jnp.full((n_agents), wheel_diameter),
        speed_mul=jnp.full((n_agents), speed_mul),
        theta_mul=jnp.full((n_agents), theta_mul),
        prox_dist_max=jnp.full((n_agents), prox_dist_max),
        proxs_cos_min=jnp.full((n_agents), prox_cos_min),
        color=jnp.tile(string_to_rgb(color), (n_agents, 1))
    )
    

# Objects 

@dataclass
class ObjectState:
    nve_idx: util.Array  # idx in NVEState
    color: util.Array

def init_object_state(n_objects, color) -> ObjectState:
    """
    Initialize object state with given parameters
    """
    return  ObjectState(
        nve_idx=jnp.arange(n_objects, dtype=int),
        color=jnp.tile(string_to_rgb(color), (n_objects, 1))
    )


# Simulator 

@dataclass
class SimulatorState:
    idx: util.Array
    box_size: util.Array
    n_agents: util.Array
    n_objects: util.Array
    num_steps_lax: util.Array
    dt: util.Array
    freq: util.Array
    neighbor_radius: util.Array
    to_jit: util.Array
    use_fori_loop: util.Array

    @staticmethod
    def get_type(attr):
        if attr in ['idx', 'n_agents', 'n_objects', 'num_steps_lax']:
            return int
        elif attr in ['box_size', 'dt', 'freq', 'neighbor_radius']:
            return float
        elif attr in ['to_jit', 'use_fori_loop']:
            return bool
        else:
            raise ValueError()
        
def init_simulator_state(
        box_size,
        n_agents,
        n_objects,
        num_steps_lax,
        dt,
        freq,
        neighbor_radius,
        to_jit,
        use_fori_loop
        ) -> SimulatorState:
    """
    Initialize simulator state with given parameters
    """
    return SimulatorState(
        idx=jnp.array([0]), 
        box_size=jnp.array([box_size]),                             
        n_agents=jnp.array([n_agents]), 
        n_objects=jnp.array([n_objects]),
        num_steps_lax=jnp.array([num_steps_lax], dtype=int), 
        dt=jnp.array([dt], dtype=float), 
        freq=jnp.array([freq], dtype=float),
        neighbor_radius=jnp.array([neighbor_radius], dtype=float),
        # Use 1*bool to transform True to 1 and False to 0
        to_jit= jnp.array([1*to_jit]), 
        use_fori_loop=jnp.array([1*use_fori_loop]))

@dataclass
class State:
    simulator_state: SimulatorState
    nve_state: NVEState
    agent_state: AgentState
    object_state: ObjectState

    def field(self, stype_or_nested_fields):
        if isinstance(stype_or_nested_fields, StateType):
            name = stype_or_nested_fields.name.lower()
            nested_fields = (f'{name}_state', )
        else:
            nested_fields = stype_or_nested_fields

        res = self
        for f in nested_fields:
            res = getattr(res, f)

        return res

    # Should we keep this function because it is duplicated below ? 
    # def nve_idx(self, etype):
    #     cond = self.e_cond(etype)
    #     return compress(range(len(cond)), cond)  # https://stackoverflow.com/questions/21448225/getting-indices-of-true-values-in-a-boolean-list

    def nve_idx(self, etype, entity_idx):
        return self.field(etype).nve_idx[entity_idx]

    def e_idx(self, etype):
        return self.nve_state.entity_idx[self.nve_state.entity_type == etype.value]

    def e_cond(self, etype):
        return self.nve_state.entity_type == etype.value

    def row_idx(self, field, nve_idx):
        return nve_idx if field == 'nve_state' else self.nve_state.entity_idx[jnp.array(nve_idx)]

    def __getattr__(self, name):
        def wrapper(e_type):
            value = getattr(self.nve_state, name)
            if isinstance(value, rigid_body.RigidBody):
                return rigid_body.RigidBody(center=value.center[self.e_cond(e_type)],
                                            orientation=value.orientation[self.e_cond(e_type)])
            else:
                return value[self.e_cond(e_type)]
        return wrapper
    
def init_state(
        simulator_state,
        agents_state,
        objects_state,
        nve_state
        ) -> State:
    
    return State(
        simulator_state=simulator_state,
        agents_state=agents_state,
        objects_state=objects_state,
        nve_state=nve_state
    )
    
    

    