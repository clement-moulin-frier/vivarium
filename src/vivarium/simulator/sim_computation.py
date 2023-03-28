import jax.numpy as jnp
from jax import ops, vmap


@vmap
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




def sensor(displ, theta, dist_max, cos_min, neighbors):
    proxs = ops.segment_max(sensor_fn(displ, theta, dist_max, cos_min), neighbors.idx[0], len(neighbors.reference_position))
    return proxs

proxs_dist_max = box_size
proxs_cos_min = 0.

def lr_2_fwd_rot(left_spd, right_spd, wheel_diameter, base_lenght):
    fwd = (wheel_diameter / 4.) * (left_spd + right_spd)
    rot = 0.5 * (wheel_diameter / base_lenght) * (right_spd - left_spd)
    return fwd, rot

def fwd_rot_2_lr(fwd, rot, base_lenght, wheel_diameter):
    left = ((2.0 * fwd) - (rot * base_lenght)) / (wheel_diameter)
    right = ((2.0 * fwd) + (rot * base_lenght)) / (wheel_diameter)
    return left, right

def motor_command(wheel_activation):
  fwd, rot = lr_2_fwd_rot(wheel_activation[0], wheel_activation[1])
  return fwd, rot


def normal(theta):
  return jnp.array([jnp.cos(theta), jnp.sin(theta)])

normal = vmap(normal)


def cross(array):
  return jnp.hstack((array[:, -1:], array[:, :1]))


def dynamics(shift, displacement, map_dim, speed_mul=1., theta_mul=1., dt=1e-1):
  def move(boids, fwd, rot):
    R, theta, *_ = boids
    n = normal(theta)
    return (shift(R, dt * speed_mul * n * jnp.tile(fwd, (map_dim, 1)).T),
            theta + dt * rot * theta_mul)

  @jit
  def update(_, entity_state, neighbors, behavior):

    state, neighbors, external_motors = state_and_neighbors_and_motors
    boids = entity_state

    neighbors = neighbors.update(boids.positions)

    senders, receivers = neighbors.idx
    Ra = boids.positions[senders]
    Rb = boids.positions[receivers]

    dR = - space.map_bond(displacement)(Ra, Rb) # Looks like it should be opposite, but don't understand why

    proxs = sensor(dR, boids.thetas[senders], neighbors)

    #motors = cross(proxs) # Braitenberg simple

    fwd, rot = behavior(proxs, external_motors)
    print('after beh')

    state[EntityType.PREY.name] = Population(*move(boids, fwd, rot), jnp.array(EntityType.PREY.value))

    return state, neighbors, external_motors

  return update

update = dynamics(dt=1e-1)

def run():
    global state, neighbors, motor_input
    print('start run')
    boids_buffer = []

    while True:
        if not sim_start:
            continue

        new_state, neighbors, _ = lax.fori_loop(0, num_lax_loops, update, (state, neighbors, motor_input))

        # If the neighbor list can't fit in the allocation, rebuild it but bigger.
        if neighbors.did_buffer_overflow:
            print('REBUILDING')
            neighbors = neighbor_fn.allocate(state[EntityType.PREY.name].R)
            state, neighbors = lax.fori_loop(0, 50, update, (state, neighbors))
            assert not neighbors.did_buffer_overflow
        else:
            state = new_state

        boids_buffer += [state[EntityType.PREY.name]]
    print('stop run')
    return "test" #np.array(state).tolist()