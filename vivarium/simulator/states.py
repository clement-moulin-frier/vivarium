from typing import Optional, List, Union
from enum import Enum

import matplotlib.colors as mcolors
import jax.numpy as jnp 

from jax import random
from jax_md import util, simulate, rigid_body
from jax_md.dataclasses import dataclass
from jax_md.rigid_body import RigidBody


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
    

# NVE (we could potentially rename it entities ? What do you think ?) 
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


@dataclass
class ObjectState:
    nve_idx: util.Array  # idx in NVEState
    color: util.Array


# TODO : I think it would make more sense to have max_agents, max_objects here instead of n_***
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

    # TODO : Should we keep this function because it is duplicated below ? 
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


# Helper function to transform a color string into rgb with matplotlib colors
def _string_to_rgb(color_str):
    return jnp.array(list(mcolors.to_rgb(color_str)))


def init_simulator_state(
        box_size: float = 100.,
        n_agents: int = 10,
        n_objects: int = 2,
        num_steps_lax: int = 4,
        dt: float = 0.1,
        freq: float = 40.,
        neighbor_radius: float = 100.,
        to_jit: bool = True,
        use_fori_loop: bool = False
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


def _init_positions(key_pos, positions, n_elements, box_size, n_dims=2):
    assert (len(positions) == n_elements if positions else True)
    # If positions are passed, transform them in jax array
    if positions:
        positions = jnp.array(positions)
    # Else initialize random positions
    else:
        positions = random.uniform(key_pos, (n_elements, n_dims)) * box_size
    return positions

def _init_existing(n_existing, n_elements):
    if n_existing:
        assert n_existing <= n_elements
        existing_arr = jnp.ones((n_existing))
        non_existing_arr = jnp.zeros((n_elements - n_existing))
        exists_array = jnp.concatenate((existing_arr, non_existing_arr))
    else:
        exists_array = jnp.ones((n_elements))
    return exists_array

# TODO : Should also add union float, list[float] for friction, diameter ..
# Also think it would be easier to only handle the nve state in the physics engine. (if it doesn't make the  simulation run slower)
# Here it makes a really long function, and it isn't very modular 
# Plus we store different attributes of entities (e.g Agents positions and sensors) in different dataclasses
def init_nve_state(
        simulator_state: SimulatorState,
        diameter: float = 5.,
        friction: float = 0.1,
        mass_center: float = 1.,
        mass_orientation: float = 0.125,
        agents_positions: Optional[Union[List[float], None]] = None,
        objects_positions: Optional[Union[List[float], None]] = None,
        existing_agents: Optional[Union[int, List[float], None]] = None,
        existing_objects: Optional[Union[int, List[float], None]] = None,
        seed: int = 0,
        ) -> NVEState:
    """
    Initialize nve state with given parameters
    """
    n_agents = simulator_state.n_agents[0]
    n_objects = simulator_state.n_objects[0]
    n_entities = n_agents + n_objects

    key = random.PRNGKey(seed)
    key_pos, key_or = random.split(key)
    key_ag, key_obj = random.split(key_pos)

    # If we have a list of agents or objects positions, transform it into a jax array, else initialize random positions
    agents_positions = _init_positions(key_ag, agents_positions, n_agents, simulator_state.box_size)
    objects_positions = _init_positions(key_obj, objects_positions, n_objects, simulator_state.box_size)
    # Assign their positions to each entities
    positions = jnp.concatenate((agents_positions, objects_positions))

    # Assign random orientations between 0 and 2*pi
    orientations = random.uniform(key_or, (n_entities,)) * 2 * jnp.pi

    agents_entities = jnp.full(n_agents, EntityType.AGENT.value)
    object_entities = jnp.full(n_objects, EntityType.OBJECT.value)
    entity_types = jnp.concatenate((agents_entities, object_entities), dtype=int)

    existing_agents = _init_existing(existing_agents, n_agents)
    existing_objects = _init_existing(existing_objects, n_objects)
    exists = jnp.concatenate((existing_agents, existing_objects), dtype=int)

    # TODO: Why is momentum set to none ?
    return NVEState(
        position=RigidBody(center=positions, orientation=orientations),
        momentum=None,
        force=RigidBody(center=jnp.zeros((n_entities, 2)), orientation=jnp.zeros(n_entities)),
        mass=RigidBody(center=jnp.full((n_entities, 1), mass_center), orientation=jnp.full((n_entities), mass_orientation)),
        entity_type=entity_types,
        entity_idx = jnp.array(list(range(n_agents)) + list(range(n_objects))),
        diameter=jnp.full((n_entities), diameter),
        friction=jnp.full((n_entities), friction),
        exists=exists
        )


# Could implement it as a static or class method
def init_agent_state(
        simulator_state: SimulatorState,
        behavior: int = 1,
        wheel_diameter: float = 2.,
        speed_mul: float = 1.,
        theta_mul: float = 1.,
        prox_dist_max: float = 40.,
        prox_cos_min: float = 0.,
        color: str = "blue"
        ) -> AgentState:
    """
    Initialize agent state with given parameters
    """
    n_agents = simulator_state.n_agents[0]
    # TODO : Allow to define custom list of behaviors, wheel_diameters ... (in fact for all parameters)
    # if the shape if just 1 value, assign it to all agents
    # else, ensure you are given a list of arguments of size n_agents and transform it into a jax array

    return AgentState(
        nve_idx=jnp.arange(n_agents, dtype=int),
        prox=jnp.zeros((n_agents, 2)),
        motor=jnp.zeros((n_agents, 2)),
        behavior=jnp.full((n_agents), behavior),
        wheel_diameter=jnp.full((n_agents), wheel_diameter),
        speed_mul=jnp.full((n_agents), speed_mul),
        theta_mul=jnp.full((n_agents), theta_mul),
        proxs_dist_max=jnp.full((n_agents), prox_dist_max),
        proxs_cos_min=jnp.full((n_agents), prox_cos_min),
        color=jnp.tile(_string_to_rgb(color), (n_agents, 1))
    )


def init_object_state(
        simulator_state: SimulatorState,
        color: str = "red"
        ) -> ObjectState:
    """
    Initialize object state with given parameters
    """
    n_agents, n_objects = simulator_state.n_agents[0], simulator_state.n_objects[0]
    start_idx, stop_idx = n_agents, n_agents + n_objects
    objects_nve_idx = jnp.arange(start_idx, stop_idx, dtype=int)
    return  ObjectState(
        nve_idx=objects_nve_idx,
        color=jnp.tile(_string_to_rgb(color), (n_objects, 1))
    )


def init_state(
        simulator_state: SimulatorState,
        agents_state: AgentState,
        objects_state: ObjectState,
        nve_state: NVEState
        ) -> State:
  
    return State(
        simulator_state=simulator_state,
        agent_state=agents_state,
        object_state=objects_state,
        nve_state=nve_state
    )
    