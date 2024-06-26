from enum import Enum
from typing import Optional, List, Union

import inspect
import yaml
import matplotlib.colors as mcolors
import jax.numpy as jnp

from jax import random
from jax_md import util, simulate, rigid_body
from jax_md.dataclasses import dataclass
from jax_md.rigid_body import RigidBody


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


# No need to define position, momentum, force, and mass (i.e already in jax_md.simulate.NVEState)
@dataclass
class EntityState(simulate.NVEState):
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
    ent_idx: util.Array  # idx in EntityState
    prox: util.Array
    motor: util.Array
    proximity_map_dist: util.Array
    proximity_map_theta: util.Array
    behavior: util.Array
    wheel_diameter: util.Array
    speed_mul: util.Array
    max_speed: util.Array
    theta_mul: util.Array
    proxs_dist_max: util.Array
    proxs_cos_min: util.Array
    color: util.Array


@dataclass
class ObjectState:
    ent_idx: util.Array  # idx in EntityState
    color: util.Array


@dataclass
class SimulatorState:
    idx: util.Array
    box_size: util.Array
    max_agents: util.Array
    max_objects: util.Array
    num_steps_lax: util.Array
    dt: util.Array
    freq: util.Array
    neighbor_radius: util.Array
    to_jit: util.Array
    use_fori_loop: util.Array
    collision_alpha: util.Array
    collision_eps: util.Array

    @staticmethod
    def get_type(attr):
        if attr in ['idx', 'max_agents', 'max_objects', 'num_steps_lax']:
            return int
        elif attr in ['box_size', 'dt', 'freq', 'neighbor_radius', 'collision_alpha', 'collision_eps']:
            return float
        elif attr in ['to_jit', 'use_fori_loop']:
            return bool
        else:
            raise ValueError(f"Unknown attribute {attr}")
     

@dataclass
class State:
    simulator_state: SimulatorState
    entity_state: EntityState
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

    def ent_idx(self, etype, entity_idx):
        return self.field(etype).ent_idx[entity_idx]

    def e_idx(self, etype):
        return self.entity_state.entity_idx[self.entity_state.entity_type == etype.value]

    def e_cond(self, etype):
        return self.entity_state.entity_type == etype.value

    def row_idx(self, field, ent_idx):
        return ent_idx if field == 'entity_state' else self.entity_state.entity_idx[jnp.array(ent_idx)]

    def __getattr__(self, name):
        def wrapper(e_type):
            value = getattr(self.entity_state, name)
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
        max_agents: int = 10,
        max_objects: int = 2,
        num_steps_lax: int = 4,
        dt: float = 0.1,
        freq: float = 40.,
        neighbor_radius: float = 100.,
        to_jit: bool = True,
        use_fori_loop: bool = False,
        collision_alpha: float = 0.5,
        collision_eps: float = 0.1
        ) -> SimulatorState:
    """
    Initialize simulator state with given parameters
    """
    return SimulatorState(
        idx=jnp.array([0]),
        box_size=jnp.array([box_size]),              
        max_agents=jnp.array([max_agents]),
        max_objects=jnp.array([max_objects]),
        num_steps_lax=jnp.array([num_steps_lax], dtype=int),
        dt=jnp.array([dt], dtype=float),
        freq=jnp.array([freq], dtype=float),
        neighbor_radius=jnp.array([neighbor_radius], dtype=float),
        # Use 1*bool to transform True to 1 and False to 0
        to_jit= jnp.array([1*to_jit]),
        use_fori_loop=jnp.array([1*use_fori_loop]),
        collision_alpha=jnp.array([collision_alpha]),
        collision_eps=jnp.array([collision_eps]))


def _init_positions(key_pos, positions, n_entities, box_size, n_dims=2):
    assert (positions is None or len(positions) == n_entities)
    # If positions are passed, transform them in jax array
    if positions:
        positions = jnp.array(positions)
    # Else initialize random positions
    else:
        positions = random.uniform(key_pos, (n_entities, n_dims)) * box_size
    return positions

def _init_existing(n_existing, n_entities):
    # Here n_entities = max_agents or max_objects
    if n_existing:
        assert n_existing <= n_entities
        existing_arr = jnp.ones((n_existing))
        non_existing_arr = jnp.zeros((n_entities - n_existing))
        exists_array = jnp.concatenate((existing_arr, non_existing_arr))
    else:
        exists_array = jnp.ones((n_entities))
    return exists_array


# TODO : Add options to have either 1 value or a list for parameters such as diameter, friction ...
def init_entity_state(
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
        ) -> EntityState:
    """
    Initialize entities state with given parameters
    """
    max_agents = simulator_state.max_agents[0]
    max_objects = simulator_state.max_objects[0]
    n_entities = max_agents + max_objects

    key = random.PRNGKey(seed)
    key, key_agents_pos, key_objects_pos, key_orientations = random.split(key, 4)

    # If we have a list of agents or objects positions, transform it into a jax array, else initialize random positions
    agents_positions = _init_positions(key_agents_pos, agents_positions, max_agents, simulator_state.box_size)
    objects_positions = _init_positions(key_objects_pos, objects_positions, max_objects, simulator_state.box_size)
    # Assign their positions to each entities
    positions = jnp.concatenate((agents_positions, objects_positions))

    # Assign random orientations between 0 and 2*pi
    orientations = random.uniform(key_orientations, (n_entities,)) * 2 * jnp.pi

    agents_entities = jnp.full(max_agents, EntityType.AGENT.value)
    object_entities = jnp.full(max_objects, EntityType.OBJECT.value)
    entity_types = jnp.concatenate((agents_entities, object_entities), dtype=int)

    existing_agents = _init_existing(existing_agents, max_agents)
    existing_objects = _init_existing(existing_objects, max_objects)
    exists = jnp.concatenate((existing_agents, existing_objects), dtype=int)

    return EntityState(
        position=RigidBody(center=positions, orientation=orientations),
        momentum=None,
        force=RigidBody(center=jnp.zeros((n_entities, 2)), orientation=jnp.zeros(n_entities)),
        mass=RigidBody(center=jnp.full((n_entities, 1), mass_center), orientation=jnp.full((n_entities), mass_orientation)),
        entity_type=entity_types,
        entity_idx = jnp.array(list(range(max_agents)) + list(range(max_objects))),
        diameter=jnp.full((n_entities), diameter),
        friction=jnp.full((n_entities), friction),
        exists=exists
        )


def init_agent_state(
        simulator_state: SimulatorState,
        behavior: int = 1,
        wheel_diameter: float = 2.,
        speed_mul: float = 1.,
        max_speed: float = 10.,
        theta_mul: float = 1.,
        prox_dist_max: float = 40.,
        prox_cos_min: float = 0.,
        color: str = "blue"
        ) -> AgentState:
    """
    Initialize agent state with given parameters
    """
    max_agents = simulator_state.max_agents[0]

    return AgentState(
        ent_idx=jnp.arange(max_agents, dtype=int),
        prox=jnp.zeros((max_agents, 2)),
        motor=jnp.zeros((max_agents, 2)),
        proximity_map_dist=jnp.zeros((max_agents, 1)),
        proximity_map_theta=jnp.zeros((max_agents, 1)),
        behavior=jnp.full((max_agents), behavior),
        wheel_diameter=jnp.full((max_agents), wheel_diameter),
        speed_mul=jnp.full((max_agents), speed_mul),
        max_speed=jnp.full((max_agents), max_speed),
        theta_mul=jnp.full((max_agents), theta_mul),
        proxs_dist_max=jnp.full((max_agents), prox_dist_max),
        proxs_cos_min=jnp.full((max_agents), prox_cos_min),
        color=jnp.tile(_string_to_rgb(color), (max_agents, 1))
    )


def init_object_state(
        simulator_state: SimulatorState,
        color: str = "red"
        ) -> ObjectState:
    """
    Initialize object state with given parameters
    """
    max_agents, max_objects = simulator_state.max_agents[0], simulator_state.max_objects[0]
    start_idx, stop_idx = max_agents, max_agents + max_objects
    objects_nve_idx = jnp.arange(start_idx, stop_idx, dtype=int)
    return  ObjectState(
        ent_idx=objects_nve_idx,
        color=jnp.tile(_string_to_rgb(color), (max_objects, 1))
    )


def _init_state(
        simulator_state: SimulatorState,
        agents_state: AgentState,
        objects_state: ObjectState,
        entity_state: EntityState
        ) -> State:
  
    return State(
        simulator_state=simulator_state,
        agent_state=agents_state,
        object_state=objects_state,
        entity_state=entity_state
    )

def init_state(args=None):
    # Use default parameters of functions if user didn't provide input
    if not args:
        args = {}

    simulator_args = args.get('simulator', {})
    agents_args = args.get('agents', {})
    objects_args = args.get('objects', {})
    entities_args = args.get('entities', {})

    simulator_state = init_simulator_state(**simulator_args)
    agents_state = init_agent_state(simulator_state=simulator_state, **agents_args)
    objects_state = init_object_state(simulator_state=simulator_state, **objects_args)
    entity_state = init_entity_state(simulator_state=simulator_state, **entities_args)

    state = _init_state(
        simulator_state=simulator_state,
        agents_state=agents_state,
        objects_state=objects_state,
        entity_state=entity_state
        )
    
    return state


def generate_default_config_files():
    """
    Generate a default yaml file with all the default arguments in the init_params_fns (see dict below)
    """
    init_params_fns = {
        'simulator': init_simulator_state,
        'entities': init_entity_state,
        'agents': init_agent_state,
        'objects': init_object_state
    }

    # TODO : Find a way to keep the order in the config_dict (atm ordered by alphebetical order in the yaml file)
    config_dict = {}
    for parameter_name, init_parameter_fn in init_params_fns.items():
        func_sig = inspect.signature(init_parameter_fn)
        default_args = {param.name: param.default for param in func_sig.parameters.values() if param.default is not inspect._empty}
        config_dict[parameter_name] = default_args

    # Add a blank line dumper to have a cleaner yaml file
    class BlankLineDumper(yaml.SafeDumper):
        # inspired by https://stackoverflow.com/a/44284819/3786245
        def write_line_break(self, data=None):
            super().write_line_break(data)

            if len(self.indents) == 1:
                super().write_line_break()

    yaml_str = yaml.dump(config_dict, Dumper=BlankLineDumper, default_flow_style=False)

    with open('conf/scene/default.yaml', 'w') as f:
        f.write(yaml_str)
