from enum import Enum
from collections import defaultdict

import numpy as np
import jax.numpy as jnp
import matplotlib.colors as mcolors

from jax import random
from jax_md.rigid_body import RigidBody

from vivarium.environments.braitenberg.behaviors import Behaviors, behavior_to_params
from vivarium.utils.scene_configs import load_default_config
from vivarium.environments.braitenberg.selective_sensing.classes import State, AgentState, ObjectState, EntityState, EntityType


CONFIG = load_default_config()

### Helper functions to generate elements of sub states

# Helper function to transform a color string into rgb with matplotlib colors
def _string_to_rgb_array(color_str):
    return jnp.array(list(mcolors.to_rgb(color_str)))

# Helper functions to define behaviors of agents in selecting sensing case
def define_behavior_map(behavior, sensed_mask):
    """Create a dict with behavior value, params and sensed mask for a given behavior

    :param behavior: behavior value
    :param sensed_mask: list of sensed mask
    :return: params associated to the behavior
    """
    params = behavior_to_params(behavior)
    sensed_mask = jnp.array([sensed_mask])

    behavior_map = {
        'behavior': behavior,
        'params': params,
        'sensed_mask': sensed_mask
    }
    return behavior_map

def stack_behaviors(behaviors_dict_list):
    """Return a dict with the stacked information from different behaviors, params and sensed mask

    :param behaviors_dict_list: list of dicts containing behavior, params and sensed mask for 1 behavior
    :return: stacked_behavior_map
    """
    # init variables
    n_behaviors = len(behaviors_dict_list)
    sensed_length = behaviors_dict_list[0]['sensed_mask'].shape[1]

    params = np.zeros((n_behaviors, 2, 3)) # (2, 3) = params.shape
    sensed_mask = np.zeros((n_behaviors, sensed_length))
    behaviors = np.zeros((n_behaviors,))

    # iterate in the list of behaviors and update params and mask
    for i in range(n_behaviors):
        assert behaviors_dict_list[i]['sensed_mask'].shape[1] == sensed_length
        params[i] = behaviors_dict_list[i]['params']
        sensed_mask[i] = behaviors_dict_list[i]['sensed_mask']
        behaviors[i] = behaviors_dict_list[i]['behavior']

    stacked_behavior_map = {
        'behaviors': behaviors,
        'params': params,
        'sensed_mask': sensed_mask
    }

    return stacked_behavior_map

def get_agents_params_and_sensed_arr(agents_stacked_behaviors_list):
    """Generate the behaviors, params and sensed arrays in jax from a list of stacked behaviors

    :param agents_stacked_behaviors_list: list of stacked behaviors
    :return: params, sensed, behaviors
    """
    n_agents = len(agents_stacked_behaviors_list)
    params_shape = agents_stacked_behaviors_list[0]['params'].shape
    sensed_shape = agents_stacked_behaviors_list[0]['sensed_mask'].shape
    behaviors_shape = agents_stacked_behaviors_list[0]['behaviors'].shape
    # Init arrays w right shapes
    params = np.zeros((n_agents, *params_shape))
    sensed = np.zeros((n_agents, *sensed_shape))
    behaviors = np.zeros((n_agents, *behaviors_shape))

    for i in range(n_agents):
        assert agents_stacked_behaviors_list[i]['params'].shape == params_shape
        assert agents_stacked_behaviors_list[i]['sensed_mask'].shape == sensed_shape
        assert agents_stacked_behaviors_list[i]['behaviors'].shape == behaviors_shape
        params[i] = agents_stacked_behaviors_list[i]['params']
        sensed[i] = agents_stacked_behaviors_list[i]['sensed_mask']
        behaviors[i] = agents_stacked_behaviors_list[i]['behaviors']

    params = jnp.array(params)
    sensed = jnp.array(sensed)
    behaviors = jnp.array(behaviors)

    return params, sensed, behaviors

def get_positions(positions, n, box_size):
    if positions is None:
        return [None] * n
    assert len(positions) == n, f"The number of positions: {len(positions)} must match the number of entities: {n}"
    for pos in positions:
        assert len(pos) == 2, f"You have to provide position with 2 coordinates, {pos} has {len(pos)}"
        assert (min(pos) > 0 and max(pos) < box_size), f"Coordinates must be floats between 0 and box_size: {box_size}, found coordinates = {pos}"
    return positions

def check_position_redundancies(agents_pos, objects_pos):
    positions = agents_pos + objects_pos
    position_indices = defaultdict(list)

    for idx, position in enumerate(positions):
        if position is not None:
            position_indices[tuple(position)].append(idx)

    redundant_positions = {position: indices for position, indices in position_indices.items() if len(indices) > 1}

    return redundant_positions if (len(redundant_positions) > 0) else False

def get_exists(exists, n):
    if exists is None:
        return [1] * n
    assert isinstance(exists, int) and (exists < n), f"Exists must be an int inferior than {n}, {exists} is not"
    exists = [1] * exists + [None] * (n - exists)
    return exists

def set_to_none_if_all_none(lst):
    if not any(element is not None for element in lst):
        return None
    return lst


### Helper functions to generate elements sub states of the state

def init_entities(
    max_agents,
    max_objects,
    ent_sub_types,
    n_dims=CONFIG.n_dims,
    box_size=CONFIG.box_size,
    existing_agents=None,
    existing_objects=None,
    mass_center=CONFIG.mass_center,
    mass_orientation=CONFIG.mass_orientation,
    diameter=CONFIG.diameter,
    friction=CONFIG.friction,
    agents_pos=None,
    objects_pos=None,
    key_agents_pos=random.PRNGKey(CONFIG.seed),
    key_objects_pos=random.PRNGKey(CONFIG.seed+1),
    key_orientations=random.PRNGKey(CONFIG.seed+2)
):
    """Init the sub entities state (field of state)"""
    n_entities = max_agents + max_objects # we store the entities data in jax arrays of length max_agents + max_objects 
    # Assign random positions to each entity in the environment
    agents_positions = random.uniform(key_agents_pos, (max_agents, n_dims)) * box_size
    objects_positions = random.uniform(key_objects_pos, (max_objects, n_dims)) * box_size

    # TODO cet aprem
    # Replace random positions with predefined ones if they exist:
    if agents_pos:
        defined_pos = jnp.array([p if p is not None else [-1, -1] for p in agents_pos])
        mask = defined_pos[:, 0] != -1
        agents_positions = jnp.where(mask[:, None], defined_pos, agents_positions)

    if objects_pos:
        defined_pos = jnp.array([p if p is not None else [-1, -1] for p in objects_pos])
        mask = defined_pos[:, 0] != -1
        objects_positions = jnp.where(mask[:, None], defined_pos, objects_positions)

    positions = jnp.concatenate((agents_positions, objects_positions))

    # Assign random orientations between 0 and 2*pi to each entity
    orientations = random.uniform(key_orientations, (n_entities,)) * 2 * jnp.pi 

    # Assign types to the entities
    agents_entities = jnp.full(max_agents, EntityType.AGENT.value)
    object_entities = jnp.full(max_objects, EntityType.OBJECT.value)
    entity_types = jnp.concatenate((agents_entities, object_entities), dtype=int)

    # Define arrays with existing entities
    exists_agents = jnp.ones((max_agents))
    exists_objects = jnp.ones((max_objects))

    if isinstance(diameter, int) or isinstance(diameter, float):
        diameter = jnp.full((n_entities), diameter)
    else:
        assert len(diameter) == n_entities, f"Length of diameter array must be equal to the number of entities: {n_entities}"
        diameter = jnp.array(diameter)

    if existing_agents is not None:
        mask = jnp.array([e if e is not None else 0 for e in existing_agents])
        exists_agents = jnp.where(mask != 0, 1, 0)

    if existing_objects is not None:
        mask = jnp.array([e if e is not None else 0 for e in existing_objects])
        exists_objects = jnp.where(mask != 0, 1, 0)

    exists = jnp.concatenate((exists_agents, exists_objects), dtype=int)

    # Works because dictionaries are ordered in Python
    ent_subtypes = np.zeros(n_entities)
    cur_idx = 0
    for subtype_id, n_subtype in ent_sub_types.values():
        ent_subtypes[cur_idx:cur_idx+n_subtype] = subtype_id
        cur_idx += n_subtype
    ent_subtypes = jnp.array(ent_subtypes, dtype=int) 

    return EntityState(
        position=RigidBody(center=positions, orientation=orientations),
        momentum=None,
        force=RigidBody(center=jnp.zeros((n_entities, 2)), orientation=jnp.zeros(n_entities)),
        mass=RigidBody(center=jnp.full((n_entities, 1), mass_center), orientation=jnp.full((n_entities), mass_orientation)),
        entity_type=entity_types,
        ent_subtype=ent_subtypes,
        entity_idx = jnp.array(list(range(max_agents)) + list(range(max_objects))),
        diameter=diameter,
        friction=jnp.full((n_entities), friction),
        exists=exists
    )

def init_agents(
    max_agents,
    max_objects,
    params,
    sensed,
    behaviors,
    agents_color,
    wheel_diameter=CONFIG.wheel_diameter,
    speed_mul=CONFIG.speed_mul,
    max_speed=CONFIG.max_speed,
    theta_mul=CONFIG.theta_mul,
    prox_dist_max=CONFIG.prox_dist_max,
    prox_cos_min=CONFIG.prox_cos_min
):
    """Init the sub agents state (field of state)"""
    # transform prox_dist_max into a jnp array
    if isinstance(prox_dist_max, int) or isinstance(prox_dist_max, float):
        proxs_dist_max=jnp.full((max_agents), prox_dist_max)
    elif isinstance(prox_dist_max, list):
        proxs_dist_max=jnp.array(prox_dist_max)
    # transform prox_cos_min into a jnp array
    if isinstance(prox_cos_min, int) or isinstance(prox_cos_min, float):
        proxs_cos_min=jnp.full((max_agents), prox_cos_min)
    elif isinstance(prox_cos_min, list):
        proxs_cos_min=jnp.array(prox_cos_min)
    # transform wheel diameter into a jnp array
    if isinstance(wheel_diameter, int) or isinstance(wheel_diameter, float):
        wheel_diameters=jnp.full((max_agents), wheel_diameter)
    elif isinstance(wheel_diameter, list):
        wheel_diameters=jnp.array(wheel_diameter)
    return AgentState(
        # idx in the entities (ent_idx) state to map agents information in the different data structures
        ent_idx=jnp.arange(max_agents, dtype=int), 
        prox=jnp.zeros((max_agents, 2), dtype=float),
        prox_sensed_ent_type=jnp.zeros((max_agents, 2), dtype=int),
        prox_sensed_ent_idx=jnp.zeros((max_agents, 2), dtype=int),
        motor=jnp.zeros((max_agents, 2)),
        behavior=behaviors,
        params=params,
        sensed=sensed,
        wheel_diameter=wheel_diameters,
        speed_mul=jnp.full((max_agents), speed_mul),
        max_speed=jnp.full((max_agents), max_speed),
        theta_mul=jnp.full((max_agents), theta_mul),
        proxs_dist_max=proxs_dist_max,
        proxs_cos_min=proxs_cos_min,
        # Change shape of these maps so they stay constant (jax.lax.scan problem otherwise)
        proximity_map_dist=jnp.zeros((max_agents, max_agents + max_objects)),
        proximity_map_theta=jnp.zeros((max_agents, max_agents + max_objects)),
        color=agents_color
    )

def init_objects(
    max_agents,
    max_objects,
    objects_color
):
    """Init the sub objects state (field of state)"""
    start_idx, stop_idx = max_agents, max_agents + max_objects 
    objects_ent_idx = jnp.arange(start_idx, stop_idx, dtype=int)

    return ObjectState(
        ent_idx=objects_ent_idx,
        color=objects_color
    )


def init_complete_state(
    entities,
    agents,
    objects,
    max_agents,
    max_objects,
    total_ent_sub_types,
    box_size=CONFIG.box_size,
    neighbor_radius=CONFIG.neighbor_radius,
    collision_alpha=CONFIG.collision_alpha,
    collision_eps=CONFIG.collision_eps,
    dt=CONFIG.dt,
):
    """Init the complete state"""
    neighbor_radius = box_size if neighbor_radius == "None" else neighbor_radius
    return  State(
        time=0,
        dt=dt,
        box_size=box_size,
        max_agents=max_agents,
        max_objects=max_objects,
        neighbor_radius=neighbor_radius,
        collision_alpha=collision_alpha,
        collision_eps=collision_eps,
        entities=entities,
        agents=agents,
        objects=objects,
        ent_sub_types=total_ent_sub_types
    )   


def process_entity(data, box_size):
    n = data['num']
    color_str = data['color']
    color = _string_to_rgb_array(color_str)
    positions = get_positions(data.get('positions'), n, box_size)
    exists = get_exists(data.get('existing'), n)
    diameter = data.get('diameter', CONFIG.diameter) # add default diamter if not provided
    diameter_lst = [diameter] * n
    return {'n': n, 'color': color}, positions, exists, diameter_lst


def init_state(
    entities_data=CONFIG.entities_data,
    box_size=CONFIG.box_size,
    dt=CONFIG.dt,
    neighbor_radius=CONFIG.neighbor_radius,
    collision_alpha=CONFIG.collision_alpha,
    collision_eps=CONFIG.collision_eps,
    n_dims=CONFIG.n_dims,
    seed=CONFIG.seed,
    diameter=CONFIG.diameter,
    friction=CONFIG.friction,
    mass_center=CONFIG.mass_center,
    mass_orientation=CONFIG.mass_orientation,
    existing_agents=None,
    existing_objects=None,
    wheel_diameter=CONFIG.wheel_diameter,
    speed_mul=CONFIG.speed_mul,
    max_speed=CONFIG.max_speed,
    theta_mul=CONFIG.theta_mul,
    prox_dist_max=CONFIG.prox_dist_max,
    prox_cos_min=CONFIG.prox_cos_min,
) -> State:
    """ Init the jax state of the simulation from classical python / yaml scene arguments """
    key = random.PRNGKey(seed)
    key, key_agents_pos, key_objects_pos, key_orientations = random.split(key, 4)
    
    # create an enum for entities subtypes
    ent_sub_types = entities_data['EntitySubTypes']
    ent_sub_types_enum = Enum('ent_sub_types_enum', {ent_sub_types[i]: i for i in range(len(ent_sub_types))}) 
    ent_data = entities_data['Entities']

    # create max agents and max objects
    max_agents = 0
    max_objects = 0 

    # create agent and objects dictionaries 
    agents_data = {}
    objects_data = {}

    # create agents and objects attributes lists
    agents_pos = []
    agents_exist = []
    agents_proxs_dist_max = []
    agents_proxs_cos_min = []
    agents_wheel_diameter = []
    objects_pos = []
    objects_exist = []
    diameters = []

    # TODO : clean this part of the function to encapsulate the behavior and proximeter data in another helper fn
    # iterate over the entities subtypes
    for ent_sub_type in ent_sub_types:
        # get their data in the ent_data
        if ent_sub_type not in ent_data:
            raise ValueError(f"Entity subtype '{ent_sub_type}' not found in the entities data. Please select entities among {ent_sub_types}")
        data = ent_data[ent_sub_type]
        entity_data, positions, exists, diameter_lst = process_entity(data, box_size)
        diameters.extend(diameter_lst)
        
        # Check if the entity is an agent or an object
        if data['type'] == 'AGENT':
            # handle proximeter infos
            prox_dist_max_val = data['prox_dist_max'] if 'prox_dist_max' in data else prox_dist_max
            prox_cos_min_val = data['prox_cos_min'] if 'prox_cos_min' in data else prox_cos_min
            wheel_diameter_val = data['wheel_diameter'] if 'wheel_diameter' in data else wheel_diameter
            assert prox_cos_min_val < 1.0, f"prox_cos_min must be inferior to 1.0, {prox_cos_min_val} is not"
            # handle behaviors
            behavior_list = []
            # create a behavior list for all behaviors of the agent
            for beh_name, behavior_data in data['selective_behaviors'].items():
                beh_name = behavior_data['beh']
                behavior_id = Behaviors[beh_name].value
                # Init an empty mask
                sensed_mask = np.zeros((len(ent_sub_types, )))
                for sensed_type in behavior_data['sensed']:
                    try:
                        # Iteratively update it with specific sensed values
                        sensed_id = ent_sub_types_enum[sensed_type].value
                        sensed_mask[sensed_id] = 1
                    except KeyError:
                        raise ValueError(f"Unknown sensed_type '{sensed_type}' encountered in sensed entities for {ent_sub_type}. Please select entities among {ent_sub_types}")
                beh = define_behavior_map(behavior_id, sensed_mask)
                behavior_list.append(beh)
            # stack the elements of the behavior list and update the agents_data dictionary
            stacked_behaviors = stack_behaviors(behavior_list)
            entity_data['stacked_behs'] = stacked_behaviors
            agents_data[ent_sub_type] = entity_data
            agents_pos.extend(positions)
            agents_exist.extend(exists)
            agents_proxs_dist_max.extend([prox_dist_max_val] * entity_data['n'])
            agents_proxs_cos_min.extend([prox_cos_min_val] * entity_data['n'])
            agents_wheel_diameter.extend([wheel_diameter_val] * entity_data['n'])
            
            max_agents += entity_data['n']

        # only updated object counters and color if entity is an object
        elif data['type'] == 'OBJECT':
            objects_data[ent_sub_type] = entity_data
            objects_pos.extend(positions)
            objects_exist.extend(exists)
            max_objects += entity_data['n']

    redundant_positions = check_position_redundancies(agents_pos, objects_pos)
    if redundant_positions:
        raise ValueError(f"Collision detected at positions: {list(redundant_positions.keys())}")

    # Set positions to None lists if they don't contain any positions 
    agents_pos = set_to_none_if_all_none(agents_pos)
    objects_pos = set_to_none_if_all_none(objects_pos)
    agents_exist = set_to_none_if_all_none(agents_exist)
    objects_exist = set_to_none_if_all_none(objects_exist)

    # Create the params, sensed, behaviors and colors arrays 
    ag_colors_list = []
    agents_stacked_behaviors_list = []
    total_ent_sub_types = {}
    # iterate over agent types
    for agent_type, data in agents_data.items():
        n = data['n']
        stacked_behavior = data['stacked_behs']
        n_stacked_behavior = list([stacked_behavior] * n)
        tiled_color = list(np.tile(data['color'], (n, 1)))
        # update the lists with behaviors and color elements
        agents_stacked_behaviors_list = agents_stacked_behaviors_list + n_stacked_behavior
        ag_colors_list = ag_colors_list + tiled_color
        total_ent_sub_types[agent_type] = (ent_sub_types_enum[agent_type].value, n)

    # create the final jnp arrays
    agents_colors = jnp.concatenate(jnp.array([ag_colors_list]), axis=0)
    params, sensed, behaviors = get_agents_params_and_sensed_arr(agents_stacked_behaviors_list)

    # do the same for objects colors
    obj_colors_list = []
    # iterate over object types
    for objecy_type, data in objects_data.items():
        n = data['n']
        tiled_color = list(np.tile(data['color'], (n, 1)))
        obj_colors_list = obj_colors_list + tiled_color
        total_ent_sub_types[objecy_type] = (ent_sub_types_enum[objecy_type].value, n)

    objects_colors = jnp.concatenate(jnp.array([obj_colors_list]), axis=0)
    # print(total_ent_sub_types)

    # Init sub states and total state
    entities = init_entities(
        max_agents=max_agents,
        max_objects=max_objects,
        ent_sub_types=total_ent_sub_types,
        n_dims=n_dims,
        box_size=box_size,
        existing_agents=agents_exist,
        existing_objects=objects_exist,
        mass_center=mass_center,
        mass_orientation=mass_orientation,
        diameter=diameters,
        friction=friction,
        agents_pos=agents_pos,
        objects_pos=objects_pos,
        key_agents_pos=key_agents_pos,
        key_objects_pos=key_objects_pos,
        key_orientations=key_orientations
    )

    agents = init_agents(
        max_agents=max_agents,
        max_objects=max_objects,
        params=params,
        sensed=sensed,
        behaviors=behaviors,
        agents_color=agents_colors,
        wheel_diameter=agents_wheel_diameter,
        speed_mul=speed_mul,
        max_speed=max_speed,
        theta_mul=theta_mul,
        prox_dist_max=agents_proxs_dist_max,
        prox_cos_min=agents_proxs_cos_min
    )

    objects = init_objects(
        max_agents=max_agents,
        max_objects=max_objects,
        objects_color=objects_colors
    )

    state = init_complete_state(
        entities=entities,
        agents=agents,
        objects=objects,
        max_agents=max_agents,
        max_objects=max_objects,
        total_ent_sub_types=total_ent_sub_types,
        box_size=box_size,
        neighbor_radius=neighbor_radius,
        collision_alpha=collision_alpha,
        collision_eps=collision_eps,
        dt=dt
    )

    return state

