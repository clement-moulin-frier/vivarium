import jax.numpy as jnp
from jax import ops, vmap, jit
import jax
from jax_md import space, energy, rigid_body, util, simulate

from collections import namedtuple
from dataclasses import dataclass
from functools import partial
f32 = util.f32


Population = namedtuple('Population', ['position', 'theta', 'prox', 'motor', 'entity_type'])

@dataclass
class RigidRobot:
    center: util.Array
    orientation: util.Array
    prox: util.Array
    motor: util.Array
    entity_type: int
    def __getitem__(self, idx):
        return RigidRobot(self.center[idx], self.orientation[idx], self.prox[idx], self.motor[idx], self.entity_type)
    def to_rigid_body(self):
        return rigid_body.RigidBody(center=self.center, orientation=self.orientation)


def get_verlet_force_fn(engine_config, simulation_config, agent_config):
    displacement = simulation_config.displacement
    shift = simulation_config.shift
    map_dim = simulation_config.map_dim
    n_agents = simulation_config.n_agents
    dt = f32(simulation_config.dt)
    speed_mul = agent_config.speed_mul
    theta_mul = agent_config.theta_mul
    proxs_dist_max = f32(agent_config.proxs_dist_max)
    proxs_cos_min = f32(agent_config.proxs_cos_min)
    base_length = f32(agent_config.base_length)
    wheel_diameter = f32(agent_config.wheel_diameter)
    entity_behaviors = jnp.array(simulation_config.entity_behaviors, dtype=int)
    behavior_bank = engine_config.behavior_bank
    coll_force_fn = quantity.force(partial(total_collision_energy, base_length=base_length, displacement=displacement,
                                   epsilon=10, alpha=12))

    motor = jnp.zeros((simulation_config.n_agents, 2), dtype=f32)  # Should be from manual input normally
    def force_fn(state, neighbor):
        if state is None:
            return rigid_body.RigidBody(center=jnp.zeros((n_agents, map_dim)), orientation=jnp.zeros(n_agents))
        body = state.position
        senders, receivers = neighbor.idx
        Ra = body.center[senders]
        Rb = body.center[receivers]
        dR = - space.map_bond(displacement)(Ra, Rb) # Looks like it should be opposite, but don't understand why
        proxs = sensor(dR, body.orientation[senders], proxs_dist_max, proxs_cos_min, neighbor)
        motors = multi_switch(entity_behaviors, behavior_bank, proxs, motor)
        fwd, rot = motor_command(motors, base_length, wheel_diameter)
        n = normal(body.orientation)
        cur_vel = state.momentum.center / state.mass.center
        # print(jnp.max(jnp.linalg.norm(cur_vel)))
        # if jnp.max(jnp.linalg.norm(cur_vel)) > 10.:
        #     print('high speed')
        cur_fwd_vel = vmap(jnp.dot)(cur_vel, n)
        cur_rot_vel = state.momentum.orientation / state.mass.orientation
        fwd_delta = fwd - cur_fwd_vel
        rot_delta = rot - cur_rot_vel
        # motor_fwd_force = n * jnp.tile(jnp.where(fwd_delta >= 0, 1., -1.), (map_dim, 1)).T
        motor_fwd_force = f32(1e-1) * n * jnp.tile(fwd_delta, (map_dim, 1)).T
        fricton_fwd_force = - f32(1e-1) * cur_vel
        fwd_force = coll_force_fn(body.center, neighbor=neighbor) + motor_fwd_force + fricton_fwd_force
        rot_force = f32(1e-2) * rot_delta
        # if rot_force[0] > 4 or rot_force[0] < - 4:
        #     print('rot_force')
        return rigid_body.RigidBody(fwd_force, rot_force)

    return force_fn



# util.register_custom_simulation_type(RigidRobot)
def rigid_verlet_init_step(force_fn, shift_fn, dt, **sim_kwargs):
    def init_fn(key, R, kT=0., mass=f32(1.0), **kwargs):
        #bodies = R.to_rigid_body()
        force = force_fn(None, **kwargs)
        state = simulate.NVEState(R, None, force, mass)
        state = simulate.canonicalize_mass(state)
        return simulate.initialize_momenta(state, key, kT)
    def my_velocity_verlet(force_fn, shift_fn, dt, state, **kwargs):
      """Apply a single step of velocity Verlet integration to a state."""
      dt = f32(dt)
      dt_2 = f32(dt / 2)

      state = simulate.momentum_step(state, dt_2)
      state = simulate.position_step(state, shift_fn, dt, **kwargs)
      state = state.set(force=force_fn(state, **kwargs))
      state = simulate.momentum_step(state, dt_2)

      return state

    def step_fn(state, **kwargs):
        # _dt = kwargs.pop('dt', dt)

        return my_velocity_verlet(force_fn, shift_fn, dt, state, **kwargs)

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


sensor_fn = vmap(sensor_fn, (0, 0, None, None))


def sensor(displ, theta, dist_max, cos_min, neighbors):
    proxs = ops.segment_max(sensor_fn(displ, theta, dist_max, cos_min), neighbors.idx[0], len(neighbors.reference_position))
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


motor_command = vmap(motor_command, (0, None, None))


def normal(theta):
  return jnp.array([jnp.cos(theta), jnp.sin(theta)])


normal = vmap(normal)

multi_switch = jax.vmap(jax.lax.switch, (0, None, 0, 0))

import jax.numpy as jnp
from jax_md import energy, quantity
from jax import lax

# strength = 0.01

def collision_energy(displ, base_length, epsilon, alpha, **kwargs):
  # distance = jnp.linalg.norm(ag1 - ag2)
  return energy.soft_sphere(jnp.linalg.norm(displ), sigma=base_length * f32(2.), epsilon=epsilon, alpha=alpha)

collision_energy = vmap(collision_energy, (0, None, None, None))



def total_collision_energy(positions, base_length, neighbor, displacement, epsilon=1e-2, alpha=2, **kwargs):
    lax.stop_gradient(base_length)
    senders, receivers = neighbor.idx
    Ra = positions[senders]
    Rb = positions[receivers]
    dR = -space.map_bond(displacement)(Ra, Rb)
    e = collision_energy(dR, base_length, epsilon, alpha)
    #e = ops.segment_sum(e, neighbors.idx[0], len(neighbors.reference_position))
    # print('e.shape', e.shape)
    return jnp.sum(e)

def dynamics(engine_config, simulation_config, agent_config):

    displacement = simulation_config.displacement
    shift = simulation_config.shift
    map_dim = simulation_config.map_dim
    dt = simulation_config.dt

    speed_mul = agent_config.speed_mul
    theta_mul = agent_config.theta_mul
    proxs_dist_max = agent_config.proxs_dist_max
    proxs_cos_min = agent_config.proxs_cos_min
    base_length = agent_config.base_length
    wheel_diameter = agent_config.wheel_diameter

    entity_behaviors = jnp.array(simulation_config.entity_behaviors, dtype=int)
    behavior_bank = engine_config.behavior_bank

    def move(positions, thetas, fwd, rot, dpos):
        n = normal(thetas)
        return (shift(positions, dt * (dpos + speed_mul * n * jnp.tile(fwd, (map_dim, 1)).T)),
                thetas + dt * rot * theta_mul)

    def update(_, state_neighbors):

        print("update")
        state, neighs = state_neighbors

        neighbors = neighs.update(state.position)

        senders, receivers = neighbors.idx
        Ra = state.position[senders]
        Rb = state.position[receivers]

        dR = - space.map_bond(displacement)(Ra, Rb) # Looks like it should be opposite, but don't understand why

        proxs = sensor(dR, state.theta[senders], proxs_dist_max, proxs_cos_min, neighbors)

        motors = multi_switch(entity_behaviors, behavior_bank, proxs, state.motor)

        fwd, rot = motor_command(motors, base_length, wheel_diameter)

        dpos = quantity.force(total_collision_energy)(state.position, base_length, neighbors, displacement)

        new_entity_state = Population(*move(state.position, state.theta, fwd, rot, dpos=dpos), prox=proxs, motor=motors,
                                      entity_type=state.entity_type)

        return new_entity_state, neighs

    return update
