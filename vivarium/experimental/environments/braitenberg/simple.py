import logging as lg

from enum import Enum
from functools import partial
from typing import Tuple

import numpy as np
import jax.numpy as jnp

from jax import vmap, jit
from jax import random, ops, lax

from flax import struct
from jax_md.rigid_body import RigidBody
from jax_md import simulate 
from jax_md import space, rigid_body, partition, quantity

from vivarium.experimental.environments.utils import normal, distance, relative_position 
from vivarium.experimental.environments.base_env import BaseState, BaseEnv
from vivarium.experimental.environments.physics_engine import total_collision_energy, friction_force, dynamics_fn


### Define the constants and the classes of the environment to store its state ###
SPACE_NDIMS = 2

class EntityType(Enum):
    AGENT = 0
    OBJECT = 1

# Already incorporates position, momentum, force, mass and velocity
@struct.dataclass
class EntityState(simulate.NVEState):
    entity_type: jnp.array
    entity_idx: jnp.array
    diameter: jnp.array
    friction: jnp.array
    exists: jnp.array
    
@struct.dataclass
class ParticleState:
    ent_idx: jnp.array
    color: jnp.array

@struct.dataclass
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

@struct.dataclass
class ObjectState(ParticleState):
    pass

@struct.dataclass
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


### Define helper functions used to step from one state to the next one ###

#--- 1 Functions to compute the proximeter of braitenberg agents ---#
proximity_map = vmap(relative_position, (0, 0))

def sensor_fn(dist, relative_theta, dist_max, cos_min, target_exists):
    """
    Compute the proximeter activations (left, right) induced by the presence of an entity
    :param dist: distance from the agent to the entity
    :param relative_theta: angle of the entity in the reference frame of the agent (front direction at angle 0)
    :param dist_max: Max distance of the proximiter (will return 0. above this distance)
    :param cos_min: Field of view as a cosinus (e.g. cos_min = 0 means a pi/4 FoV on each proximeter, so pi/2 in total)
    :return: left and right proximeter activation in a jnp array with shape (2,)
    """
    cos_dir = jnp.cos(relative_theta)
    prox = 1. - (dist / dist_max)
    in_view = jnp.logical_and(dist < dist_max, cos_dir > cos_min)
    at_left = jnp.logical_and(True, jnp.sin(relative_theta) >= 0)
    left = in_view * at_left * prox
    right = in_view * (1. - at_left) * prox
    return jnp.array([left, right]) * target_exists  # i.e. 0 if target does not exist

sensor_fn = vmap(sensor_fn, (0, 0, 0, 0, 0))

def sensor(dist, relative_theta, dist_max, cos_min, max_agents, senders, target_exists):
    """Return the sensor values of all agents

    :param dist: relative distances between agents and targets
    :param relative_theta: relative angles between agents and targets
    :param dist_max: maximum range of proximeters
    :param cos_min: cosinus of proximeters angles
    :param max_agents: number of agents
    :param senders: indexes of agents sensing the environment
    :param target_exists: mask to indicate which sensed entities exist or not 
    :return: proximeter activations
    """
    raw_proxs = sensor_fn(dist, relative_theta, dist_max, cos_min, target_exists)
    # Computes the maximum within the proximeter activations of agents on all their neigbhors.
    proxs = ops.segment_max(
        raw_proxs,
        senders,    
        max_agents)
    
    return proxs

def compute_prox(state, agents_neighs_idx, target_exists_mask, displacement):
    """
    Set agents' proximeter activations
    :param state: full simulation State
    :param agents_neighs_idx: Neighbor representation, where sources are only agents. Matrix of shape (2, n_pairs),
    where n_pairs is the number of neighbor entity pairs where sources (first row) are agent indexes.
    :param target_exists_mask: Specify which target entities exist. Vector with shape (n_entities,).
    target_exists_mask[i] is True (resp. False) if entity of index i in state.entities exists (resp. don't exist).
    :return:
    """
    body = state.entities.position
    mask = target_exists_mask[agents_neighs_idx[1, :]]   
    senders, receivers = agents_neighs_idx
    Ra = body.center[senders]
    Rb = body.center[receivers]
    dR = - space.map_bond(displacement)(Ra, Rb)  # Looks like it should be opposite, but don't understand why

    # Create distance and angle maps between entities
    dist, theta = proximity_map(dR, body.orientation[senders])
    proximity_map_dist = jnp.zeros((state.agents.ent_idx.shape[0], state.entities.entity_idx.shape[0]))
    proximity_map_dist = proximity_map_dist.at[senders, receivers].set(dist)
    proximity_map_theta = jnp.zeros((state.agents.ent_idx.shape[0], state.entities.entity_idx.shape[0]))
    proximity_map_theta = proximity_map_theta.at[senders, receivers].set(theta)

    prox = sensor(dist, theta, state.agents.proxs_dist_max[senders],
                    state.agents.proxs_cos_min[senders], len(state.agents.ent_idx), senders, mask)
    
    return prox, proximity_map_dist, proximity_map_theta


#--- 2 Functions to compute the motor activations of braitenberg agents ---#
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
}

def behavior_to_params(behavior):
    """Return the params associated to a behavior.

    :param behavior: behavior id (int)
    :return: params
    """
    return behavior_params[behavior]

def linear_behavior(proxs, params):
    """Compute the activation of motors with a linear combination of proximeters and parameters

    :param proxs: proximeter values of an agent
    :param params: parameters of an agent (mapping proxs to motor values)
    :return: motor values
    """
    return params.dot(jnp.hstack((proxs, 1.)))

v_linear_behavior = vmap(linear_behavior, in_axes=(0, 0))
                    
def compute_motor(proxs, params, behaviors, motors):
    """Compute new motor values. If behavior is manual, keep same motor values. Else, compute new values with proximeters and params.

    :param proxs: proximeters of all agents
    :param params: parameters mapping proximeters to new motor values
    :param behaviors: array of behaviors
    :param motors: current motor values
    :return: new motor values
    """
    manual = jnp.where(behaviors == Behaviors.MANUAL.value, 1, 0)
    manual_mask = jnp.broadcast_to(jnp.expand_dims(manual, axis=1), motors.shape)
    linear_motor_values = v_linear_behavior(proxs, params)
    motor_values = linear_motor_values * (1 - manual_mask) + motors * manual_mask
    return motor_values

def lr_2_fwd_rot(left_spd, right_spd, base_length, wheel_diameter):
    """Return the forward and angular speeds according the the speeds of left and right wheels

    :param left_spd: left wheel speed
    :param right_spd: right wheel speed
    :param base_length: distance between two wheels (diameter of the agent)
    :param wheel_diameter: diameter of wheels
    :return: forward and angular speeds
    """
    fwd = (wheel_diameter / 4.) * (left_spd + right_spd)
    rot = 0.5 * (wheel_diameter / base_length) * (right_spd - left_spd)
    return fwd, rot

def fwd_rot_2_lr(fwd, rot, base_length, wheel_diameter):
    """Return the left and right wheels speeds according to the forward and angular speeds

    :param fwd: forward speed
    :param rot: angular speed
    :param base_length: distance between wheels (diameter of agent)
    :param wheel_diameter: diameter of wheels
    :return: left wheel speed, right wheel speed
    """
    left = ((2.0 * fwd) - (rot * base_length)) / wheel_diameter
    right = ((2.0 * fwd) + (rot * base_length)) / wheel_diameter
    return left, right

def motor_command(wheel_activation, base_length, wheel_diameter):
    """Return the forward and angular speed according to wheels speeds

    :param wheel_activation: wheels speeds
    :param base_length: distance between wheels
    :param wheel_diameter: wheel diameters
    :return: forward and angular speeds
    """
    fwd, rot = lr_2_fwd_rot(wheel_activation[0], wheel_activation[1], base_length, wheel_diameter)
    return fwd, rot

motor_command = vmap(motor_command, (0, 0, 0))


#--- 3 Functions to compute the different forces in the environment ---#
# TODO : Refactor the code in order to simply the definition of a total force fn incorporating different forces
def braintenberg_force_fn(displacement):
    """Return the force function of the environment

    :param displacement: displacement function to compute distances between entities
    :return: force function
    """
    coll_force_fn = quantity.force(partial(total_collision_energy, displacement=displacement))

    def collision_force(state, neighbor, exists_mask):
        """Returns the collision force function of the environment

        :param state: state
        :param neighbor: neighbor maps of entities
        :param exists_mask: mask on existing entities
        :return: collision force function
        """
        return coll_force_fn(
            state.entities.position.center,
            neighbor=neighbor,
            exists_mask=exists_mask,
            diameter=state.entities.diameter,
            epsilon=state.collision_eps,
            alpha=state.collision_alpha
            )

    def motor_force(state, exists_mask):
        """Returns the motor force function of the environment

        :param state: state
        :param exists_mask: mask on existing entities
        :return: motor force function
        """
        agent_idx = state.agents.ent_idx

        body = rigid_body.RigidBody(
            center=state.entities.position.center[agent_idx],
            orientation=state.entities.position.orientation[agent_idx]
            )
        
        n = normal(body.orientation)

        fwd, rot = motor_command(
            state.agents.motor,
            state.entities.diameter[agent_idx],
            state.agents.wheel_diameter
            )
        # `a_max` arg is deprecated in recent versions of jax, replaced by `max`
        fwd = jnp.clip(fwd, a_max=state.agents.max_speed)

        cur_vel = state.entities.momentum.center[agent_idx] / state.entities.mass.center[agent_idx]
        cur_fwd_vel = vmap(jnp.dot)(cur_vel, n)
        cur_rot_vel = state.entities.momentum.orientation[agent_idx] / state.entities.mass.orientation[agent_idx]
        
        fwd_delta = fwd - cur_fwd_vel
        rot_delta = rot - cur_rot_vel

        fwd_force = n * jnp.tile(fwd_delta, (SPACE_NDIMS, 1)).T * jnp.tile(state.agents.speed_mul, (SPACE_NDIMS, 1)).T
        rot_force = rot_delta * state.agents.theta_mul

        center=jnp.zeros_like(state.entities.position.center).at[agent_idx].set(fwd_force)
        orientation=jnp.zeros_like(state.entities.position.orientation).at[agent_idx].set(rot_force)

        # apply mask to make non existing agents stand still
        orientation = jnp.where(exists_mask, orientation, 0.)
        # Because position has SPACE_NDMS dims, need to stack the mask to give it the same shape as center
        exists_mask = jnp.stack([exists_mask] * SPACE_NDIMS, axis=1)
        center = jnp.where(exists_mask, center, 0.)

        return rigid_body.RigidBody(center=center,
                                    orientation=orientation)
    
    def force_fn(state, neighbor, exists_mask):
        """Returns the total force applied on the environment

        :param state: state
        :param neighbor: neighbor map
        :param exists_mask: existing entities mask
        :return: total force
        """
        mf = motor_force(state, exists_mask)
        cf = collision_force(state, neighbor, exists_mask)
        ff = friction_force(state, exists_mask)
        
        center = cf + ff + mf.center
        orientation = mf.orientation
        return rigid_body.RigidBody(center=center, orientation=orientation)

    return force_fn

#--- 4 Define the environment class with its different functions (step ...) ---#
class BraitenbergEnv(BaseEnv):
    def __init__(self, state, seed=42):
        self.seed = seed
        self.init_key = random.PRNGKey(seed)
        self.displacement, self.shift = space.periodic(state.box_size)
        self.init_fn, self.apply_physics = dynamics_fn(self.displacement, self.shift, braintenberg_force_fn)
        self.neighbor_fn = partition.neighbor_list(
            self.displacement, 
            state.box_size,
            r_cutoff=state.neighbor_radius,
            dr_threshold=10.,
            capacity_multiplier=1.5,
            format=partition.Sparse
        )

        self.neighbors, self.agents_neighs_idx = self.allocate_neighbors(state)

    def distance(self, point1, point2):
            return distance(self.displacement, point1, point2)
    
    @partial(jit, static_argnums=(0,))
    def _step(self, state: State, neighbors: jnp.array, agents_neighs_idx: jnp.array) -> Tuple[State, jnp.array]:
        # 1 : Compute agents proximeter
        exists_mask = jnp.where(state.entities.exists == 1, 1, 0)
        prox, proximity_dist_map, proximity_dist_theta = compute_prox(state, agents_neighs_idx, target_exists_mask=exists_mask, displacement=self.displacement)

        # 2 : Compute motor activations according to new proximeter values
        motor = compute_motor(prox, state.agents.params, state.agents.behavior, state.agents.motor)
        agents = state.agents.replace(
            prox=prox, 
            proximity_map_dist=proximity_dist_map, 
            proximity_map_theta=proximity_dist_theta,
            motor=motor
        )

        # 3 : Update the state with new agents proximeter and motor values
        state = state.replace(agents=agents)

        # 4 : Move the entities by applying forces on them (collision, friction and motor forces for agents)
        entities = self.apply_physics(state, neighbors)
        state = state.replace(time=state.time+1, entities=entities)

        # 5 : Update neighbors
        neighbors = neighbors.update(state.entities.position.center)
        return state, neighbors
    
    def step(self, state: State) -> State:
        if state.entities.momentum is None:
             state = self.init_fn(state, self.init_key)
        current_state = state
        state, neighbors = self._step(current_state, self.neighbors, self.agents_neighs_idx)
        if self.neighbors.did_buffer_overflow:
            # reallocate neighbors and run the simulation from current_state
            lg.warning(f'NEIGHBORS BUFFER OVERFLOW at step {state.time}: rebuilding neighbors')
            neighbors, self.agents_neighs_idx = self.allocate_neighbors(state)
            assert not neighbors.did_buffer_overflow

        self.neighbors = neighbors
        return state

    def allocate_neighbors(self, state, position=None):
        neighbors = super().allocate_neighbors(state, position)

        # Also update the neighbor idx of agents (not the cleanest to attribute it to with self here)
        ag_idx = state.entities.entity_type[neighbors.idx[0]] == EntityType.AGENT.value
        agents_neighs_idx = neighbors.idx[:, ag_idx]
        
        return neighbors, agents_neighs_idx

#--- 5 Define helper functions to initialize a state #
def init_state(
    box_size=100,
    dt=0.1,
    max_agents=10,
    max_objects=2,
    neighbor_radius=100.,
    collision_alpha=0.5,
    collision_eps=0.1,
    n_dims=2,
    seed=0,
    diameter=5.0,
    friction=0.1,
    mass_center=1.0,
    mass_orientation=0.125,
    existing_agents=None,
    existing_objects=None,
    behavior=Behaviors.AGGRESSION.value,
    wheel_diameter=2.0,
    speed_mul=1.0,
    max_speed=10.0,
    theta_mul=1.0,
    prox_dist_max=40.0,
    prox_cos_min=0.0,
    agents_color=jnp.array([0.0, 0.0, 1.0]),
    objects_color=jnp.array([1.0, 0.0, 0.0])
) -> State:

    key = random.PRNGKey(seed)
    key, key_agents_pos, key_objects_pos, key_orientations = random.split(key, 4)

    entities = init_entities(
        max_objects=max_objects,
        max_agents=max_agents,
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
        behavior=behavior,
        wheel_diameter=wheel_diameter,
        speed_mul=speed_mul,
        max_speed=max_speed,
        theta_mul=theta_mul, 
        prox_dist_max=prox_dist_max,
        prox_cos_min=prox_cos_min,
        agents_color=agents_color

    )

    objects = init_objects(
        max_agents=max_agents,
        max_objects=max_objects,
        objects_color=objects_color
    )

    state = init_complete_state(
        entities=entities, 
        agents=agents, 
        objects=objects,
        box_size=box_size,
        max_agents=max_agents,
        max_objects=max_objects,
        neighbor_radius=neighbor_radius,
        collision_alpha=collision_alpha,
        collision_eps=collision_eps,
        dt=dt
    )

    return state

def init_entities(
    max_agents,
    max_objects, 
    n_dims, 
    box_size,
    existing_agents,
    existing_objects,
    mass_center,
    mass_orientation,
    diameter,
    friction,
    key_agents_pos, 
    key_objects_pos, 
    key_orientations
):
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

def init_agents(
    max_agents,
    behavior,
    wheel_diameter,
    speed_mul,
    max_speed,
    theta_mul,
    prox_dist_max,
    prox_cos_min,
    agents_color
):
    # Need to use a np array because jax jax array can't be the key of a dict (for fn behaviors_to_params)
    np_behaviors = np.full((max_agents), behavior)
    params = jnp.array([behavior_to_params(behavior) for behavior in np_behaviors])
    behaviors = jnp.array(np_behaviors)
    return AgentState(
        # idx in the entities (ent_idx) state to map agents information in the different data structures
        ent_idx=jnp.arange(max_agents, dtype=int), 
        prox=jnp.zeros((max_agents, 2)),
        motor=jnp.zeros((max_agents, 2)),
        behavior=behaviors,
        params=params,
        wheel_diameter=jnp.full((max_agents), wheel_diameter),
        speed_mul=jnp.full((max_agents), speed_mul),
        max_speed=jnp.full((max_agents), max_speed),
        theta_mul=jnp.full((max_agents), theta_mul),
        proxs_dist_max=jnp.full((max_agents), prox_dist_max),
        proxs_cos_min=jnp.full((max_agents), prox_cos_min),
        proximity_map_dist=jnp.zeros((max_agents, 1)),
        proximity_map_theta=jnp.zeros((max_agents, 1)),
        color=jnp.tile(agents_color, (max_agents, 1))
    )

def init_objects(
    max_agents,
    max_objects,
    objects_color
):
    # Entities idx of objects
    start_idx, stop_idx = max_agents, max_agents + max_objects 
    objects_ent_idx = jnp.arange(start_idx, stop_idx, dtype=int)

    return ObjectState(
        ent_idx=objects_ent_idx,
        color=jnp.tile(objects_color, (max_objects, 1))
    )

def init_complete_state(
    entities, 
    agents, 
    objects, 
    box_size, 
    max_agents, 
    max_objects, 
    neighbor_radius, 
    collision_alpha, 
    collision_eps, 
    dt
):
    return State(
        time=0,
        box_size=box_size,
        max_agents=max_agents,
        max_objects=max_objects,
        neighbor_radius=neighbor_radius,
        collision_alpha=collision_alpha,
        collision_eps=collision_eps,
        dt=dt,
        entities=entities,
        agents=agents,
        objects=objects
    )   
