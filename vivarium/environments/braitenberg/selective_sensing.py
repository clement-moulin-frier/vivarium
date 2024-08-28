import logging as lg

from enum import Enum
from functools import partial
from typing import Tuple

import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.colors as mcolors

from jax import vmap, jit
from jax import random, lax

from flax import struct
from jax_md.rigid_body import RigidBody
from jax_md.dataclasses import dataclass as md_dataclass
from jax_md import space, partition, simulate

from vivarium.environments.utils import distance 
from vivarium.environments.base_env import BaseState, BaseEnv
from vivarium.environments.physics_engine import dynamics_fn
from vivarium.environments.braitenberg.simple import proximity_map, sensor_fn
from vivarium.environments.braitenberg.simple import Behaviors, behavior_to_params, linear_behavior
from vivarium.environments.braitenberg.simple import braintenberg_force_fn


### Define the constants and the classes of the environment to store its state ###
SPACE_NDIMS = 2

class EntityType(Enum):
    AGENT = 0
    OBJECT = 1

# Set the class of all states to jax_md dataclass instead of struct dataclass
# What could be done in the future is to set it back to struct and simplify client server connection 

# Already incorporates position, momentum, force, mass and velocity
@md_dataclass
class EntityState(simulate.NVEState):
    entity_type: jnp.array
    ent_subtype: jnp.array
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
    sensed: jnp.array
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
    ent_sub_types: dict
    entities: EntityState
    agents: AgentState
    objects: ObjectState    

# Not part of the state but part of the environment
@md_dataclass
class Neighbors:
    neighbors: jnp.array
    agents_neighs_idx: jnp.array
    agents_idx_dense: jnp.array


# TODO : Should refactor the function to split the returns
def get_relative_displacement(state, agents_neighs_idx, displacement_fn):
    """Get all infos relative to distance and orientation between all agents and their neighbors

    :param state: state
    :param agents_neighs_idx: idx all agents neighbors
    :param displacement_fn: jax md function enabling to know the distance between points
    :return: distance array, angles array, distance map for all agents, angles map for all agents
    """
    body = state.entities.position
    senders, receivers = agents_neighs_idx
    Ra = body.center[senders]
    Rb = body.center[receivers]
    dR = - space.map_bond(displacement_fn)(Ra, Rb)  # Looks like it should be opposite, but don't understand why

    dist, theta = proximity_map(dR, body.orientation[senders])
    proximity_map_dist = jnp.zeros((state.agents.ent_idx.shape[0], state.entities.entity_idx.shape[0]))
    proximity_map_dist = proximity_map_dist.at[senders, receivers].set(dist)
    proximity_map_theta = jnp.zeros((state.agents.ent_idx.shape[0], state.entities.entity_idx.shape[0]))
    proximity_map_theta = proximity_map_theta.at[senders, receivers].set(theta)
    return dist, theta, proximity_map_dist, proximity_map_theta

# def linear_behavior(proxs, params):
#     """Compute the activation of motors with a linear combination of proximeters and parameters

#     :param proxs: proximeter values of an agent
#     :param params: parameters of an agent (mapping proxs to motor values)
#     :return: motor values
#     """
#     return params.dot(jnp.hstack((proxs, 1.)))

def compute_motor(proxs, params, behaviors, motors):
    """Compute new motor values. If behavior is manual, keep same motor values. Else, compute new values with proximeters and params.

    :param proxs: proximeters of all agents
    :param params: parameters mapping proximeters to new motor values
    :param behaviors: array of behaviors
    :param motors: current motor values
    :return: new motor values
    """
    manual = jnp.where(behaviors == Behaviors.MANUAL.value, 1, 0)
    manual_mask = manual
    linear_motor_values = linear_behavior(proxs, params)
    motor_values = linear_motor_values * (1 - manual_mask) + motors * manual_mask
    return motor_values

### 1 : Functions for selective sensing with occlusion

def update_mask(mask, left_n_right_types, ent_type):
    """Update a mask of 

    :param mask: mask that will be applied on sensors of agents
    :param left_n_right_types: types of left adn right sensed entities
    :param ent_type: entity subtype (e.g 1 for predators)
    :return: mask
    """
    cur = jnp.where(left_n_right_types == ent_type, 0, 1)
    mask *= cur
    return mask

def keep_mask(mask, left_n_right_types, ent_type):
    """Return the mask unchanged

    :param mask: mask
    :param left_n_right_types: left_n_right_types
    :param ent_type: ent_type
    :return: mask
    """
    return mask

def mask_proxs_occlusion(proxs, left_n_right_types, ent_sensed_arr):
    """Mask the proximeters of agents with occlusion

    :param proxs: proxiemters of agents without occlusion (shape = (2,))
    :param e_sensed_types: types of both entities sensed at left and right (shape=(2,))
    :param ent_sensed_arr: mask of sensed subtypes by the agent (e.g jnp.array([0, 1, 0, 1]) if sense only entities of subtype 1 and 4)
    :return: updated proximeters according to sensed_subtypes
    """
    mask = jnp.array([1, 1])
    # Iterate on the array of sensed entities mask
    for ent_type, sensed in enumerate(ent_sensed_arr):
        # If an entity is sensed, update the mask, else keep it as it is
        mask = jax.lax.cond(sensed, update_mask, keep_mask, mask, left_n_right_types, ent_type)
    # Update the mask with 0s where the mask is, else keep the prox value
    proxs = jnp.where(mask, 0, proxs)
    return proxs

# Example :
# ent_sensed_arr = jnp.array([0, 1, 0, 0, 1])
# proxs = jnp.array([0.8, 0.2])
# e_sensed_types = jnp.array([4, 4]) # Modify these values to check it works
# print(mask_proxs_occlusion(proxs, e_sensed_types, ent_sensed_arr))

def compute_behavior_motors(state, params, sensed_mask, behavior, motor, agent_proxs, sensed_ent_idx):
    """Compute the motor values for a specific behavior

    :param state: state
    :param params: behavior params params
    :param sensed_mask: sensed_mask for this behavior
    :param behavior: behavior
    :param motor: motor values
    :param agent_proxs: agent proximeters (unmasked)
    :param sensed_ent_idx: idx of left and right entities sensed 
    :return: right motor values for this behavior 
    """
    left_n_right_types = state.entities.ent_subtype[sensed_ent_idx]
    behavior_proxs = mask_proxs_occlusion(agent_proxs, left_n_right_types, sensed_mask)
    motors = compute_motor(behavior_proxs, params, behaviors=behavior, motors=motor)
    return motors

# See for the vectorizing idx because already in a vmaped function here
compute_all_behavior_motors = vmap(compute_behavior_motors, in_axes=(None, 0, 0, 0, None, None, None))


def compute_occlusion_proxs_motors(state, agent_idx, params, sensed, behaviors, motor, raw_proxs, ag_idx_dense_senders, ag_idx_dense_receivers):
    """_summary_

    :param state: state
    :param agent_idx: agent idx in entities
    :param params: params arrays for all agent's behaviors
    :param sensed: sensed mask arrays for all agent's behaviors
    :param behaviors: agent behaviors array
    :param motor: agent motors
    :param raw_proxs: raw_proximeters for all agents (shape=(n_agents * (n_entities - 1), 2))
    :param ag_idx_dense_senders: ag_idx_dense_senders to get the idx of raw proxs (shape=(2, n_agents * (n_entities - 1))
    :param ag_idx_dense_receivers: ag_idx_dense_receivers (shape=(n_agents, n_entities - 1))
    :return: _description_
    """
    behavior = jnp.expand_dims(behaviors, axis=1) 
    # Compute the neighbors idx of the agent and get its raw proximeters (of shape (n_entities -1 , 2))
    ent_ag_neighs_idx = ag_idx_dense_senders[agent_idx]
    agent_raw_proxs = raw_proxs[ent_ag_neighs_idx]

    # Get the max and arg max of these proximeters on axis 0, gives results of shape (2,)
    agent_proxs = jnp.max(agent_raw_proxs, axis=0)
    argmax = jnp.argmax(agent_raw_proxs, axis=0)
    # Get the real entity idx of the left and right sensed entities from dense neighborhoods
    sensed_ent_idx = ag_idx_dense_receivers[agent_idx][argmax]
    
    # Compute the motor values for all behaviors and do a mean on it
    motor_values = compute_all_behavior_motors(state, params, sensed, behavior, motor, agent_proxs, sensed_ent_idx)
    motors = jnp.mean(motor_values, axis=0)

    return agent_proxs, motors

compute_all_agents_proxs_motors_occl = vmap(compute_occlusion_proxs_motors, in_axes=(None, 0, 0, 0, 0, 0, None, None, None))


### 2 : Functions for selective sensing without occlusion

def mask_sensors(state, agent_raw_proxs, ent_type_id, ent_neighbors_idx):
    """Mask the raw proximeters of agents for a specific entity type 

    :param state: state
    :param agent_raw_proxs: raw_proximeters of agent (shape=(n_entities - 1), 2)
    :param ent_type_id: entity subtype id (e.g 0 for PREYS)
    :param ent_neighbors_idx: idx of agent neighbors in entities arrays
    :return: updated agent raw proximeters
    """
    mask = jnp.where(state.entities.ent_subtype[ent_neighbors_idx] == ent_type_id, 0, 1)
    mask = jnp.expand_dims(mask, 1)
    mask = jnp.broadcast_to(mask, agent_raw_proxs.shape)
    return agent_raw_proxs * mask

def dont_change(state, agent_raw_proxs, ent_type_id, ent_neighbors_idx):
    """Leave the agent raw_proximeters unchanged

    :param state: state
    :param agent_raw_proxs: agent_raw_proxs
    :param ent_type_id: ent_type_id
    :param ent_neighbors_idx: ent_neighbors_idx
    :return: agent_raw_proxs
    """
    return agent_raw_proxs

def compute_behavior_prox(state, agent_raw_proxs, ent_neighbors_idx, sensed_entities):
    """Compute the proximeters for a specific behavior

    :param state: state
    :param agent_raw_proxs: agent raw proximeters
    :param ent_neighbors_idx: idx of agent neighbors
    :param sensed_entities: array of sensed entities
    :return: updated proximeters
    """
    # iterate over all the types in sensed_entities and return if they are sensed or not
    for ent_type_id, sensed in enumerate(sensed_entities):
        # change the proxs if you don't perceive the entity, else leave them unchanged
        agent_raw_proxs = lax.cond(sensed, dont_change, mask_sensors, state, agent_raw_proxs, ent_type_id, ent_neighbors_idx)
    # Compute the final proxs with a max on the updated raw_proxs
    proxs = jnp.max(agent_raw_proxs, axis=0)
    return proxs

def compute_behavior_proxs_motors(state, params, sensed, behavior, motor, agent_raw_proxs, ent_neighbors_idx):
    """Return the proximeters and the motors for a specific behavior

    :param state: state
    :param params: params of the behavior
    :param sensed: sensed mask of the behavior
    :param behavior: behavior
    :param motor: motor values
    :param agent_raw_proxs: agent_raw_proxs
    :param ent_neighbors_idx: ent_neighbors_idx
    :return: behavior proximeters, behavior motors
    """
    behavior_prox = compute_behavior_prox(state, agent_raw_proxs, ent_neighbors_idx, sensed)
    behavior_motors = compute_motor(behavior_prox, params, behavior, motor)
    return behavior_prox, behavior_motors

# vmap on params, sensed and behavior (parallelize on all agents behaviors at once, but not motorrs because are the same)
compute_all_behavior_proxs_motors = vmap(compute_behavior_proxs_motors, in_axes=(None, 0, 0, 0, None, None, None))

def compute_agent_proxs_motors(state, agent_idx, params, sensed, behavior, motor, raw_proxs, ag_idx_dense_senders, ag_idx_dense_receivers):
    """Compute the agent proximeters and motors for all behaviors

    :param state: state
    :param agent_idx: idx of the agent in entities
    :param params: array of params for all behaviors
    :param sensed: array of sensed mask for all behaviors
    :param behavior: array of behaviors
    :param motor: motor values
    :param raw_proxs: raw_proximeters of all agents
    :param ag_idx_dense_senders: ag_idx_dense_senders to get the idx of raw proxs (shape=(2, n_agents * (n_entities - 1))
    :param ag_idx_dense_receivers: ag_idx_dense_receivers (shape=(n_agents, n_entities - 1))
    :return: array of agent_proximeters, mean of behavior motors
    """
    behavior = jnp.expand_dims(behavior, axis=1)
    ent_ag_idx = ag_idx_dense_senders[agent_idx]
    ent_neighbors_idx = ag_idx_dense_receivers[agent_idx]
    agent_raw_proxs = raw_proxs[ent_ag_idx]

    # vmap on params, sensed, behaviors and motorss (vmap on all agents)
    agent_proxs, agent_motors = compute_all_behavior_proxs_motors(state, params, sensed, behavior, motor, agent_raw_proxs, ent_neighbors_idx)
    mean_agent_motors = jnp.mean(agent_motors, axis=0)

    return agent_proxs, mean_agent_motors

compute_all_agents_proxs_motors = vmap(compute_agent_proxs_motors, in_axes=(None, 0, 0, 0, 0, 0, None, None, None))




class SelectiveSensorsEnv(BaseEnv):
    def __init__(self, state, occlusion=True, seed=42):
        """Init the selective sensors braitenberg env 

        :param state: simulation state already complete
        :param occlusion: wether to use sensors with occlusion or not, defaults to True
        :param seed: random seed, defaults to 42
        """
        self.seed = seed
        self.occlusion = occlusion
        self.compute_all_agents_proxs_motors = self.choose_agent_prox_motor_function()
        self.init_key = random.PRNGKey(seed)
        self.displacement, self.shift = space.periodic(state.box_size)
        self.init_fn, self.apply_physics = dynamics_fn(self.displacement, self.shift, braintenberg_force_fn)
        # Do a warning at the moment if neighbor radius is < box_size
        if state.neighbor_radius < state.box_size:
            lg.warn("Neighbor radius < Box size, this might cause problems for neighbors arrays and proximity maps updates")
        self.neighbor_fn = partition.neighbor_list(
            self.displacement, 
            state.box_size,
            r_cutoff=state.neighbor_radius,
            dr_threshold=10.,
            capacity_multiplier=1.5,
            format=partition.Sparse
        )
        self.neighbors_storage = self.allocate_neighbors(state)

    def distance(self, point1, point2):
        """Returns the distance between two points

        :param point1: point1 coordinates
        :param point2: point1 coordinates
        :return: distance between two points
        """
        return distance(self.displacement, point1, point2)
    
    # At the moment doesn't work because the _step function isn't recompiled 
    def choose_agent_prox_motor_function(self):
        """Returns the function to compute the proximeters and the motors with or without occlusion

        :return: compute_all_agents_proxs_motors function
        """
        if self.occlusion:
            prox_motor_function = compute_all_agents_proxs_motors_occl
        else:
            prox_motor_function = compute_all_agents_proxs_motors
        return prox_motor_function
    
    @partial(jit, static_argnums=(0,))
    def _step_env(self, state: State, neighbors_storage: Neighbors) -> Tuple[State, Neighbors]:
        """Do one jitted step in the environment and return the updated state, as well as updated neighbors array

        :param state: current state
        :param neighbors_storage: class storing all neighbors information
        :return: new state, neighbors storage wih updated neighbors
        """

        # Retrieve different neighbors format
        neighbors = neighbors_storage.neighbors
        agents_neighs_idx = neighbors_storage.agents_neighs_idx
        ag_idx_dense = neighbors_storage.agents_idx_dense
        senders, receivers = agents_neighs_idx
        ag_idx_dense_senders, ag_idx_dense_receivers = ag_idx_dense

        # Compute raw proxs for all agents first 
        dist, relative_theta, proximity_dist_map, proximity_dist_theta = get_relative_displacement(
            state, 
            agents_neighs_idx, 
            displacement_fn=self.displacement
        )

        dist_max = state.agents.proxs_dist_max[senders]
        cos_min = state.agents.proxs_cos_min[senders]
        # TODO : shouldn't the agents_neighs_idx[1, :] be receivers ?
        target_exist_mask = state.entities.exists[agents_neighs_idx[1, :]]
        # Compute agents raw proximeters (proximeters for all neighbors)
        raw_proxs = sensor_fn(dist, relative_theta, dist_max, cos_min, target_exist_mask)

        # Compute real agents proximeters and motors
        agent_proxs, mean_agent_motors = self.compute_all_agents_proxs_motors(
            state,
            state.agents.ent_idx,
            state.agents.params,
            state.agents.sensed,
            state.agents.behavior,
            state.agents.motor,
            raw_proxs,
            ag_idx_dense_senders,
            ag_idx_dense_receivers,
        )

        # Update agents state
        agents = state.agents.set(
            prox=agent_proxs, 
            proximity_map_dist=proximity_dist_map, 
            proximity_map_theta=proximity_dist_theta,
            motor=mean_agent_motors
        )

        # Update the entities and the state
        state = state.replace(agents=agents)
        entities = self.apply_physics(state, neighbors)
        state = state.replace(time=state.time+1, entities=entities)

        # Update the neighbors storage
        neighbors = neighbors.update(state.entities.position.center)
        neighbors_storage = Neighbors(
            neighbors=neighbors, 
            agents_neighs_idx=agents_neighs_idx, 
            agents_idx_dense=ag_idx_dense
        )

        return state, neighbors_storage
    

    @partial(jax.jit, static_argnums=(0, 3))
    def _steps(self, state, neighbor_storage, num_updates):
        lg.debug('Compile _steps function in SelectiveSensing environment')

        """Update the current state by doing a _step_env update num_updates times (this results in faster simulations) 

        :param state: _description_
        :param neighbor_storage: _description_
        :param num_updates: _description_
        """
        
        def step_fn(carry, _):
            """Apply a step function to return new state and neighbors storage in a jax.lax.scan update

            :param carry: tuple of (state, neighbors storage)
            :param _: dummy xs for jax.lax.scan
            :return: tuple of (carry, carry) with carry=(new_state, new_neighbors _sotrage)
            """
            state, neighbors_storage = carry
            new_state, new_neighbors_storage = self._step_env(state, neighbors_storage)
            carry = (new_state, new_neighbors_storage)
            return carry, carry
        
        (state, neighbor_storage), _ = jax.lax.scan(
            step_fn, 
            (state, neighbor_storage), 
            xs=None, 
            length=num_updates
        )

        return state, neighbor_storage

    
    def step(self, state: State, num_updates: int = 4) -> State:
        """Do num_updates jitted steps in the environment and return the updated state. This function also handles the neighbors mechanism and hence isn't jitted

        :param state: current state
        :param num_updates: number of jitted_steps
        :return: next state
        """
        # Because momentum is initialized to None, need to initialize it with init_fn from jax_md
        if state.entities.momentum is None:
             state = self.init_fn(state, self.init_key)
        
         # Save the first state
        current_state = state

        state, neighbors_storage = self._steps(current_state, self.neighbors_storage, num_updates) 

        # Check if neighbors buffer overflowed
        if neighbors_storage.neighbors.did_buffer_overflow:
            # reallocate neighbors and run the simulation from current_state if it is the case
            lg.warning(f'NEIGHBORS BUFFER OVERFLOW at step {state.time}: rebuilding neighbors')
            self.neighbors_storage = self.allocate_neighbors(state)
            # Because there was an error, we need to re-run this simulation loop from the copy of the current_state we created (and check wether it worked or not after)
            state, neighbors_storage = self._steps(current_state, self.neighbors_storage, num_updates) 
            assert not neighbors_storage.neighbors.did_buffer_overflow

        return state


    def allocate_neighbors(self, state, position=None):
        """Allocate the neighbors according to the state

        :param state: state
        :param position: position of entities in the state, defaults to None
        :return: Neighbors object with neighbors (sparse representation), idx of agent's neighbors, neighbors (dense representation) 
        """
        # get the sparse representation of neighbors (shape=(n_neighbors_pairs, 2))
        position = state.entities.position.center if position is None else position
        neighbors = self.neighbor_fn.allocate(position)

        # Also update the neighbor idx of agents
        ag_idx = state.entities.entity_type[neighbors.idx[0]] == EntityType.AGENT.value
        agents_neighs_idx = neighbors.idx[:, ag_idx]

        # Give the idx of the agents in sparse representation, under a dense representation (used to get the raw proxs in compute motors function)
        agents_idx_dense_senders = jnp.array([jnp.argwhere(jnp.equal(agents_neighs_idx[0, :], idx)).flatten() for idx in jnp.arange(state.max_agents)]) 
        # Note: jnp.argwhere(jnp.equal(self.agents_neighs_idx[0, :], idx)).flatten() ~ jnp.where(agents_idx[0, :] == idx)
        
        # Give the idx of the agent neighbors in dense representation
        agents_idx_dense_receivers = agents_neighs_idx[1, :][agents_idx_dense_senders]
        agents_idx_dense = agents_idx_dense_senders, agents_idx_dense_receivers

        neighbor_storage = Neighbors(neighbors=neighbors, agents_neighs_idx=agents_neighs_idx, agents_idx_dense=agents_idx_dense)
        return neighbor_storage


### Default values
seed = 0
n_dims = 2
box_size = 100
diameter = 5.0
friction = 0.1
mass_center = 1.0
mass_orientation = 0.125
# Set neighbor radius to box_size to ensure good conversion from sparse to dense neighbors
neighbor_radius = box_size
collision_alpha = 0.5
collision_eps = 0.1
dt = 0.1
wheel_diameter = 2.0
speed_mul = 1.0
max_speed = 10.0
theta_mul = 1.0
prox_dist_max = 40.0
prox_cos_min = 0.0
existing_agents = None
existing_objects = None

entities_sbutypes = ['PREYS', 'PREDS', 'RESSOURCES', 'POISON']

preys_data = {
    'type': 'AGENT',
    'num': 5,
    'color': 'blue',
    'selective_behaviors': {
        'love': {'beh': 'LOVE', 'sensed': ['PREYS', 'RESSOURCES']},
        'fear': {'beh': 'FEAR', 'sensed': ['PREDS', 'POISON']}
    }}

preds_data = {
    'type': 'AGENT',
    'num': 5,
    'color': 'red',
    'selective_behaviors': {
        'aggr': {'beh': 'AGGRESSION','sensed': ['PREYS']},
        'fear': {'beh': 'FEAR','sensed': ['POISON']
        }
    }}

ressources_data = {
    'type': 'OBJECT',
    'num': 5,
    'color': 'green'}

poison_data = {
    'type': 'OBJECT',
    'num': 5,
    'color': 'purple'}

entities_data = {
    'EntitySubTypes': entities_sbutypes,
    'Entities': {
        'PREYS': preys_data,
        'PREDS': preds_data,
        'RESSOURCES': ressources_data,
        'POISON': poison_data
    }}

### Helper functions to generate the state

# Helper function to transform a color string into rgb with matplotlib colors
def _string_to_rgb(color_str):
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

def init_entities(
    max_agents,
    max_objects,
    ent_sub_types,
    n_dims=n_dims,
    box_size=box_size,
    existing_agents=None,
    existing_objects=None,
    mass_center=mass_center,
    mass_orientation=mass_orientation,
    diameter=diameter,
    friction=friction,
    key_agents_pos=random.PRNGKey(seed),
    key_objects_pos=random.PRNGKey(seed+1),
    key_orientations=random.PRNGKey(seed+2)
):
    """Init the sub entities state"""
    existing_agents = max_agents if not existing_agents else existing_agents
    existing_objects = max_objects if not existing_objects else existing_objects

    n_entities = max_agents + max_objects # we store the entities data in jax arrays of length max_agents + max_objects 
    # Assign random positions to each entity in the environment
    agents_positions = random.uniform(key_agents_pos, (max_agents, n_dims)) * box_size
    objects_positions = random.uniform(key_objects_pos, (max_objects, n_dims)) * box_size
    positions = jnp.concatenate((agents_positions, objects_positions))
    # Assign random orientations between 0 and 2*pi to each entity
    orientations = random.uniform(key_orientations, (n_entities,)) * 2 * jnp.pi
    # Assign types to the entities
    agents_entities = jnp.full(max_agents, EntityType.AGENT.value)
    object_entities = jnp.full(max_objects, EntityType.OBJECT.value)
    entity_types = jnp.concatenate((agents_entities, object_entities), dtype=int)
    # Define arrays with existing entities
    exists_agents = jnp.concatenate((jnp.ones((existing_agents)), jnp.zeros((max_agents - existing_agents))))
    exists_objects = jnp.concatenate((jnp.ones((existing_objects)), jnp.zeros((max_objects - existing_objects))))
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
        diameter=jnp.full((n_entities), diameter),
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
    wheel_diameter=wheel_diameter,
    speed_mul=speed_mul,
    max_speed=max_speed,
    theta_mul=theta_mul,
    prox_dist_max=prox_dist_max,
    prox_cos_min=prox_cos_min
):
    """Init the sub agents state"""
    return AgentState(
        # idx in the entities (ent_idx) state to map agents information in the different data structures
        ent_idx=jnp.arange(max_agents, dtype=int), 
        prox=jnp.zeros((max_agents, 2)),
        motor=jnp.zeros((max_agents, 2)),
        behavior=behaviors,
        params=params,
        sensed=sensed,
        wheel_diameter=jnp.full((max_agents), wheel_diameter),
        speed_mul=jnp.full((max_agents), speed_mul),
        max_speed=jnp.full((max_agents), max_speed),
        theta_mul=jnp.full((max_agents), theta_mul),
        proxs_dist_max=jnp.full((max_agents), prox_dist_max),
        proxs_cos_min=jnp.full((max_agents), prox_cos_min),
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
    """Init the sub objects state"""
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
    box_size=box_size,
    neighbor_radius=neighbor_radius,
    collision_alpha=collision_alpha,
    collision_eps=collision_eps,
    dt=dt,
):
    """Init the complete state"""
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


def init_state(
    entities_data=entities_data,
    box_size=box_size,
    dt=dt,
    neighbor_radius=neighbor_radius,
    collision_alpha=collision_alpha,
    collision_eps=collision_eps,
    n_dims=n_dims,
    seed=seed,
    diameter=diameter,
    friction=friction,
    mass_center=mass_center,
    mass_orientation=mass_orientation,
    existing_agents=None,
    existing_objects=None,
    wheel_diameter=wheel_diameter,
    speed_mul=speed_mul,
    max_speed=max_speed,
    theta_mul=theta_mul,
    prox_dist_max=prox_dist_max,
    prox_cos_min=prox_cos_min,
) -> State:
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

    # iterate over the entities subtypes
    for ent_sub_type in ent_sub_types:
        # get their data in the ent_data
        data = ent_data[ent_sub_type]
        color_str = data['color']
        color = _string_to_rgb(color_str)
        n = data['num']

        # Check if the entity is an agent or an object
        if data['type'] == 'AGENT':
            max_agents += n
            behavior_list = []
            # create a behavior list for all behaviors of the agent
            for beh_name, behavior_data in data['selective_behaviors'].items():
                beh_name = behavior_data['beh']
                behavior_id = Behaviors[beh_name].value
                # Init an empty mask
                sensed_mask = np.zeros((len(ent_sub_types, )))
                for sensed_type in behavior_data['sensed']:
                    # Iteratively update it with specific sensed values
                    sensed_id = ent_sub_types_enum[sensed_type].value
                    sensed_mask[sensed_id] = 1
                beh = define_behavior_map(behavior_id, sensed_mask)
                behavior_list.append(beh)
            # stack the elements of the behavior list and update the agents_data dictionary
            stacked_behaviors = stack_behaviors(behavior_list)
            agents_data[ent_sub_type] = {'n': n, 'color': color, 'stacked_behs': stacked_behaviors}

        # only updated object counters and color if entity is an object
        elif data['type'] == 'OBJECT':
            max_objects += n
            objects_data[ent_sub_type] = {'n': n, 'color': color}

    # Create the params, sensed, behaviors and colors arrays 

    # init empty lists
    colors = []
    agents_stacked_behaviors_list = []
    total_ent_sub_types = {}
    for agent_type, data in agents_data.items():
        n = data['n']
        stacked_behavior = data['stacked_behs']
        n_stacked_behavior = list([stacked_behavior] * n)
        tiled_color = list(np.tile(data['color'], (n, 1)))
        # update the lists with behaviors and color elements
        agents_stacked_behaviors_list = agents_stacked_behaviors_list + n_stacked_behavior
        colors = colors + tiled_color
        total_ent_sub_types[agent_type] = (ent_sub_types_enum[agent_type].value, n)

    # create the final jnp arrays
    agents_colors = jnp.concatenate(jnp.array([colors]), axis=0)
    params, sensed, behaviors = get_agents_params_and_sensed_arr(agents_stacked_behaviors_list)

    # do the same for objects colors
    colors = []
    for objecy_type, data in objects_data.items():
        n = data['n']
        tiled_color = list(np.tile(data['color'], (n, 1)))
        colors = colors + tiled_color
        total_ent_sub_types[objecy_type] = (ent_sub_types_enum[objecy_type].value, n)

    objects_colors = jnp.concatenate(jnp.array([colors]), axis=0)
    # print(total_ent_sub_types)

    # Init sub states and total state
    entities = init_entities(
        max_agents=max_agents,
        max_objects=max_objects,
        ent_sub_types=total_ent_sub_types,
        n_dims=n_dims,
        box_size=box_size,
        existing_agents=existing_agents,
        existing_objects=existing_objects,
        mass_center=mass_center,
        mass_orientation=mass_orientation,
        diameter=diameter,
        friction=friction,
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
        wheel_diameter=wheel_diameter,
        speed_mul=speed_mul,
        max_speed=max_speed,
        theta_mul=theta_mul,
        prox_dist_max=prox_dist_max,
        prox_cos_min=prox_cos_min
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


if __name__ == "__main__":
    state = init_state()
    env = SelectiveSensorsEnv(state)

    env.step(state, num_updates=5)
    env.step(state, num_updates=6)
