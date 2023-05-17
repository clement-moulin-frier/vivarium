import jax
from jax import ops, vmap, lax
import jax.numpy as jnp

from jax_md import space, rigid_body, util, simulate, energy, quantity
from jax_md.dataclasses import dataclass

from functools import partial

f32 = util.f32


@dataclass
class NVEState(simulate.NVEState):
    prox: util.Array
    motor: util.Array
    behavior: util.Array
    wheel_diameter: util.Array
    base_length: util.Array
    speed_mul: util.Array
    theta_mul: util.Array
    proxs_dist_max: util.Array
    proxs_cos_min: util.Array
    entity_type: util.Array


def get_verlet_force_fn(displacement, map_dim):
    coll_force_fn = quantity.force(partial(total_collision_energy, displacement=displacement,
                                           epsilon=10., alpha=12))

    def collision_force(state, neighbor):
        return coll_force_fn(state.position.center, neighbor=neighbor, base_length=state.base_length)

    def friction_force(state, neighbor):
        cur_vel = state.momentum.center / state.mass.center
        return - f32(1e-1) * cur_vel

    def motor_force(state, neighbor):
        body = state.position
        fwd, rot = motor_command(state.motor, state.base_length, state.wheel_diameter)
        n = normal(body.orientation)
        cur_vel = state.momentum.center / state.mass.center
        cur_fwd_vel = vmap(jnp.dot)(cur_vel, n)
        cur_rot_vel = state.momentum.orientation / state.mass.orientation
        fwd_delta = fwd - cur_fwd_vel
        rot_delta = rot - cur_rot_vel
        fwd_force = f32(1e-1) * n * jnp.tile(fwd_delta, (map_dim, 1)).T
        rot_force = f32(1e-2) * rot_delta
        return rigid_body.RigidBody(center=fwd_force, orientation=rot_force)

    def force_fn(state, neighbor):
        mf = motor_force(state, neighbor)
        return rigid_body.RigidBody(center=collision_force(state, neighbor) + friction_force(state, neighbor) + mf.center,
                                    orientation=mf.orientation)

    return force_fn



# util.register_custom_simulation_type(RigidRobot)
def dynamics_rigid(displacement, shift, map_dim, dt, behavior_bank, force_fn=None, **sim_kwargs):
    force_fn = force_fn or get_verlet_force_fn(displacement, map_dim)
    shape = rigid_body.monomer
    def init_fn(key, positions, orientations, agent_configs_as_array_dict, kT=0.):
        key, subkey = jax.random.split(key)
        n_agents = positions.shape[0]
        proxs = jnp.zeros((n_agents, 2))
        motors = jnp.zeros((n_agents, 2))
        body_positions = rigid_body.RigidBody(center=positions, orientation=orientations)
        force = rigid_body.RigidBody(center=jnp.zeros((n_agents, map_dim)), orientation=jnp.zeros(n_agents))
        state = simulate.NVEState(position=body_positions, momentum=None, force=force, mass=shape.mass())
        state = simulate.canonicalize_mass(state)
        state = simulate.initialize_momenta(state, key, kT)
        return NVEState(position=state.position, momentum=state.momentum, force=state.force, mass=state.mass,
                        prox=proxs, motor=motors, **agent_configs_as_array_dict)

    def physics_fn(state, force, shift_fn, dt, neighbor):
        """Apply a single step of velocity Verlet integration to a state."""
        dt = f32(dt)
        dt_2 = f32(dt / 2)
        # state = sensorimotor(state, neighbor)  # now in step_fn
        state = simulate.momentum_step(state, dt_2)
        state = simulate.position_step(state, shift_fn, dt, neighbor=neighbor)
        state = state.set(force=force)
        state = simulate.momentum_step(state, dt_2)

        return state

    def sensorimotor(state, neighbor):
        body = state.position
        senders, receivers = neighbor.idx
        Ra = body.center[senders]
        Rb = body.center[receivers]
        dR = - space.map_bond(displacement)(Ra, Rb) # Looks like it should be opposite, but don't understand why
        prox = sensor(dR, body.orientation[senders], state.proxs_dist_max[senders], state.proxs_cos_min[senders], neighbor)
        motor = multi_switch(state.behavior, behavior_bank, prox, state.motor)

        return state.set(prox=prox, motor=motor)

    def step_fn(state, neighbor, **kwargs):
        # _dt = kwargs.pop('dt', dt)
        state = sensorimotor(state, neighbor)
        force = force_fn(state, neighbor)
        return physics_fn(state, force, shift, dt, neighbor=neighbor)

    return init_fn, step_fn


def sensor_fn(displ, theta, dist_max, cos_min):
    dist = jnp.linalg.norm(displ)
    n = jnp.array([jnp.cos( - theta), jnp.sin(- theta)])
    rot_matrix = jnp.array([[n[0], - n[1]], [n[1], n[0]]])
    rot_displ = jnp.dot(rot_matrix, jnp.reshape(displ, (2, 1))).reshape((-1, ))
    cos_dir = rot_displ[0] / dist
    prox = 1. - (dist / dist_max)
    in_view = jnp.logical_and(dist < dist_max, cos_dir > cos_min)
    at_left = jnp.logical_and(True, rot_displ[1] >= 0)
    left = in_view * at_left * prox
    right = in_view * (1. - at_left) * prox
    return jnp.array([left, right])


sensor_fn = vmap(sensor_fn, (0, 0, 0, 0))


def sensor(displ, theta, dist_max, cos_min, neighbors):
    proxs = ops.segment_max(sensor_fn(displ, theta, dist_max, cos_min),
                            neighbors.idx[0], len(neighbors.reference_position))
    return proxs


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


def normal(theta):
  return jnp.array([jnp.cos(theta), jnp.sin(theta)])


normal = vmap(normal)

multi_switch = jax.vmap(jax.lax.switch, (0, None, 0, 0))


def collision_energy(displacement_fn, r_a, r_b, l_a, l_b, epsilon, alpha):
    dist = jnp.linalg.norm(displacement_fn(r_a, r_b))
    sigma = l_a + l_b
    return energy.soft_sphere(dist, sigma=sigma, epsilon=epsilon, alpha=alpha)


collision_energy = vmap(collision_energy, (None, 0, 0, 0, 0, None, None))


def total_collision_energy(positions, base_length, neighbor, displacement, epsilon=1e-2, alpha=2, **kwargs):
    lax.stop_gradient(base_length)
    senders, receivers = neighbor.idx
    Ra = positions[senders]
    Rb = positions[receivers]
    l_a = base_length[senders]
    l_b = base_length[receivers]
    e = collision_energy(displacement, Ra, Rb, l_a, l_b, epsilon, alpha)
    return jnp.sum(e)
