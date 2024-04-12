from functools import partial

import jax
import jax.numpy as jnp

from jax import ops, vmap, lax
from jax_md import space, rigid_body, util, simulate, energy, quantity
f32 = util.f32


# Only work on 2D environments atm
SPACE_NDIMS = 2

@vmap
def normal(theta):
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])

def switch_fn(fn_list):
    def switch(index, *operands):
        return jax.lax.switch(index, fn_list, *operands)
    return switch


# Helper functions for collisions

def collision_energy(displacement_fn, r_a, r_b, l_a, l_b, epsilon, alpha, mask):
    """Compute the collision energy between a pair of particles

    :param displacement_fn: displacement function of jax_md 
    :param r_a: position of particle a 
    :param r_b: position of particle b 
    :param l_a: diameter of particle a 
    :param l_b: diameter of particle b 
    :param epsilon: interaction energy scale 
    :param alpha: interaction stiffness
    :param mask: set the energy to 0 if one of the particles is masked 
    :return: collision energy between both particles
    """
    dist = jnp.linalg.norm(displacement_fn(r_a, r_b))
    sigma = (l_a + l_b) / 2
    e = energy.soft_sphere(dist, sigma=sigma, epsilon=epsilon, alpha=f32(alpha))
    return jnp.where(mask, e, 0.)

collision_energy = vmap(collision_energy, (None, 0, 0, 0, 0, None, None, 0))


def total_collision_energy(positions, diameters, epsilons, alphas, neighbor, displacement, exists_mask):
    """Compute the collision energy between all neighboring pairs of particles in the system 

    :param positions: positions of all the particles 
    :param diameters: diameters of all the particles 
    :param neighbor: neighbor array of the system
    :param displacement: dipalcement function of jax_md
    :param exists_mask: mask to specify which particles exist 
    :param epsilon: interaction energy scale between two particles
    :param alpha: interaction stiffness between two particles
    :return: sum of all collisions energies of the system 
    """
    diameters = lax.stop_gradient(diameters)
    senders, receivers = neighbor.idx

    r_senders = positions[senders]
    r_receivers = positions[receivers]
    l_senders = diameters[senders]
    l_receivers = diameters[receivers]
    eps = epsilons[receivers]
    alph = alphas[receivers]

    # Set collision energy to zero if the sender or receiver is non existing
    mask = exists_mask[senders] * exists_mask[receivers]
    energies = collision_energy(displacement,
                                 r_senders,
                                 r_receivers,
                                 l_senders,
                                 l_receivers,
                                 eps,
                                 alph,
                                 mask)
    return jnp.sum(energies)


# Helper functions for motor function

def lr_2_fwd_rot(left_spd, right_spd, base_length, wheel_diameter):
    fwd = (wheel_diameter / 4.) * (left_spd + right_spd)
    rot = 0.5 * (wheel_diameter / base_length) * (right_spd - left_spd)
    return fwd, rot


def fwd_rot_2_lr(fwd, rot, base_length, wheel_diameter):
    left = ((2.0 * fwd) - (rot * base_length)) / wheel_diameter
    right = ((2.0 * fwd) + (rot * base_length)) / wheel_diameter
    return left, right


def motor_command(wheel_activation, base_length, wheel_diameter):
    fwd, rot = lr_2_fwd_rot(wheel_activation[0], wheel_activation[1], base_length, wheel_diameter)
    return fwd, rot

motor_command = vmap(motor_command, (0, 0, 0))


# Functions to compute the verlet force on the whole system

def get_verlet_force_fn(displacement):
    coll_force_fn = quantity.force(partial(total_collision_energy, displacement=displacement))

    def collision_force(state, neighbor, exists_mask):
        return coll_force_fn(
            state.entities_state.position.center,
            neighbor=neighbor,
            exists_mask=exists_mask,
            diameters=state.entities_state.diameter,
            epsilons=state.entities_state.collision_eps,
            alphas=state.entities_state.collision_alpha
            )

    def friction_force(state, exists_mask):
        cur_vel = state.entities_state.momentum.center / state.entities_state.mass.center
        # stack the mask to give it the same shape as cur_vel (that has 2 rows for forward and angular velocities)
        mask = jnp.stack([exists_mask] * 2, axis=1)
        cur_vel = jnp.where(mask, cur_vel, 0.)
        return - jnp.tile(state.entities_state.friction, (SPACE_NDIMS, 1)).T * cur_vel

    def motor_force(state, exists_mask):
        agent_idx = state.agent_state.nve_idx

        body = rigid_body.RigidBody(
            center=state.entities_state.position.center[agent_idx],
            orientation=state.entities_state.position.orientation[agent_idx]
            )
        
        n = normal(body.orientation)

        fwd, rot = motor_command(
            state.agent_state.motor,
            state.entities_state.diameter[agent_idx],
            state.agent_state.wheel_diameter
            )
        # `a_max` arg is deprecated in recent versions of jax, replaced by `max`
        fwd = jnp.clip(fwd, a_max=state.agent_state.max_speed)

        cur_vel = state.entities_state.momentum.center[agent_idx] / state.entities_state.mass.center[agent_idx]
        cur_fwd_vel = vmap(jnp.dot)(cur_vel, n)
        cur_rot_vel = state.entities_state.momentum.orientation[agent_idx] / state.entities_state.mass.orientation[agent_idx]
        
        fwd_delta = fwd - cur_fwd_vel
        rot_delta = rot - cur_rot_vel

        fwd_force = n * jnp.tile(fwd_delta, (SPACE_NDIMS, 1)).T * jnp.tile(state.agent_state.speed_mul, (SPACE_NDIMS, 1)).T
        rot_force = rot_delta * state.agent_state.theta_mul

        center=jnp.zeros_like(state.entities_state.position.center).at[agent_idx].set(fwd_force)
        orientation=jnp.zeros_like(state.entities_state.position.orientation).at[agent_idx].set(rot_force)

        # apply mask to make non existing agents stand still
        orientation = jnp.where(exists_mask, orientation, 0.)
        # Because position has SPACE_NDMS dims, need to stack the mask to give it the same shape as center
        exists_mask = jnp.stack([exists_mask] * SPACE_NDIMS, axis=1)
        center = jnp.where(exists_mask, center, 0.)


        return rigid_body.RigidBody(center=center,
                                    orientation=orientation)

    def force_fn(state, neighbor, exists_mask):
        mf = motor_force(state, exists_mask)
        cf = collision_force(state, neighbor, exists_mask)
        ff = friction_force(state, exists_mask)
        
        center = cf + ff + mf.center
        orientation = mf.orientation
        return rigid_body.RigidBody(center=center, orientation=orientation)

    return force_fn


# Helper functions for sensors

def dist_theta(displ, theta):
    """
    Compute the relative distance and angle from a source agent to a target agent
    :param displ: Displacement vector (jnp arrray with shape (2,) from source to target
    :param theta: Orientation of the source agent (in the reference frame of the map)
    :return: dist: distance from source to target.
    relative_theta: relative angle of the target in the reference frame of the source agent (front direction at angle 0)
    """
    dist = jnp.linalg.norm(displ)
    norm_displ = displ / dist
    theta_displ = jnp.arccos(norm_displ[0]) * jnp.sign(jnp.arcsin(norm_displ[1]))
    relative_theta = theta_displ - theta
    return dist, relative_theta

proximity_map = vmap(dist_theta, (0, 0))


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


def sensor(displ, theta, dist_max, cos_min, max_agents, senders, target_exists):
    dist, relative_theta = proximity_map(displ, theta)
    proxs = ops.segment_max(sensor_fn(dist, relative_theta, dist_max, cos_min, target_exists),
                            senders, max_agents)
    return proxs


# Functions to compute the dynamics of the whole system 

def dynamics_rigid(displacement, shift, behavior_bank, force_fn=None):
    force_fn = force_fn or get_verlet_force_fn(displacement)
    multi_switch = jax.vmap(switch_fn(behavior_bank), (0, 0, 0))
    # shape = rigid_body.monomer

    def init_fn(state, key, kT=0.):
        key, _ = jax.random.split(key)
        assert state.entities_state.momentum is None
        assert not jnp.any(state.entities_state.force.center) and not jnp.any(state.entities_state.force.orientation)

        state = state.set(entities_state=simulate.initialize_momenta(state.entities_state, key, kT))
        return state
    
    def mask_momentum(entities_state, exists_mask):
        """
        Set the momentum values to zeros for non existing entities
        :param entities_state: entities_state
        :param exists_mask: bool array specifying which entities exist or not
        :return: entities_state: new entities state state with masked momentum values
        """
        orientation = jnp.where(exists_mask, entities_state.momentum.orientation, 0)
        exists_mask = jnp.stack([exists_mask] * SPACE_NDIMS, axis=1)
        center = jnp.where(exists_mask, entities_state.momentum.center, 0)
        momentum = rigid_body.RigidBody(center=center, orientation=orientation)
        return entities_state.set(momentum=momentum)

    def physics_fn(state, force, shift_fn, dt, neighbor, mask):
        """Apply a single step of velocity Verlet integration to a state."""
        # dt = f32(dt)
        dt_2 = dt / 2.  # f32(dt / 2)
        # state = sensorimotor(state, neighbor)  # now in step_fn
        entities_state = simulate.momentum_step(state.entities_state, dt_2)
        entities_state = simulate.position_step(entities_state, shift_fn, dt, neighbor=neighbor)
        entities_state = entities_state.set(force=force)
        entities_state = simulate.momentum_step(entities_state, dt_2)
        entities_state = mask_momentum(entities_state, mask)

        return state.set(entities_state=entities_state)

    def compute_prox(state, agent_neighs_idx, target_exists_mask):
        """
        Set agents' proximeter activations
        :param state: full simulation State
        :param agent_neighs_idx: Neighbor representation, where sources are only agents. Matrix of shape (2, n_pairs),
        where n_pairs is the number of neighbor entity pairs where sources (first row) are agent indexes.
        :param target_exists_mask: Specify which target entities exist. Vector with shape (n_entities,).
        target_exists_mask[i] is True (resp. False) if entity of index i in state.entities_state exists (resp. don't exist).
        :return:
        """
        body = state.entities_state.position
        mask = target_exists_mask[agent_neighs_idx[1, :]]
        senders, receivers = agent_neighs_idx
        Ra = body.center[senders]
        Rb = body.center[receivers]
        dR = - space.map_bond(displacement)(Ra, Rb)  # Looks like it should be opposite, but don't understand why
        prox = sensor(dR, body.orientation[senders], state.agent_state.proxs_dist_max[senders],
                      state.agent_state.proxs_cos_min[senders], len(state.agent_state.nve_idx), senders, mask)
        return state.agent_state.set(prox=prox)

    def sensorimotor(agent_state):
        motor = multi_switch(agent_state.behavior, agent_state.prox, agent_state.motor)
        return agent_state.set(motor=motor)

    def step_fn(state, neighbor, agent_neighs_idx):
        exists_mask = (state.entities_state.exists == 1)  # Only existing entities have effect on others
        state = state.set(agent_state=compute_prox(state, agent_neighs_idx, target_exists_mask=exists_mask))
        state = state.set(agent_state=sensorimotor(state.agent_state))
        force = force_fn(state, neighbor, exists_mask)
        state = physics_fn(state, force, shift, state.simulator_state.dt[0], neighbor=neighbor, mask=exists_mask)
        return state

    return init_fn, step_fn
