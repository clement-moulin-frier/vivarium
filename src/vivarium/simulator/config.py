import param
from param import Parameterized, Parameter
# from vivarium.simulator.simulator import EntityType
from jax_md import space, partition
import jax
import jax.numpy as jnp
import numpy as np



class AgentConfig(Parameterized):

    wheel_diameter = param.Number(2.)
    base_length = param.Number(10.)
    speed_mul = param.Number(0.1)
    theta_mul = param.Number(0.1)
    neighbor_radius = param.Number(100., bounds=(0, None))
    proxs_dist_max = param.Number(100., bounds=(0, None))
    proxs_cos_min = param.Number(0., bounds=(-1., 1.))

    def json(self):
        return self.param.serialize_parameters()


class SimulatorConfig(Parameterized):
    box_size = param.Number(100., bounds=(0, None))
    map_dim = param.Integer(2, bounds=(1, None))
    agent_config = param.ClassSelector(AgentConfig, instantiate=False)
    num_steps_lax = param.Integer(50)
    num_lax_loops = param.Integer(1)
    dt = param.Number(0.1)
    freq = param.Number(100.)
    to_jit = param.Boolean(True)
    displacement = Parameter()
    shift = Parameter()


    def json(self):
        return self.param.serialize_parameters(subset=['box_size', 'map_dim'])

    @param.depends('box_size', watch=True, on_init=True)
    def _update_space(self):
        self.displacement, self.shift = space.periodic(self.box_size)

    @param.depends('displacement', 'box_size', 'agent_config.neighbor_radius', watch=True, on_init=True)
    def _update_neighbor_fn(self):
        self.neighbor_fn = partition.neighbor_list(self.displacement,
                                                   self.box_size,
                                                   r_cutoff=self.agent_config.neighbor_radius,
                                                   dr_threshold=10.,
                                                   capacity_multiplier=1.5,
                                                   format=partition.Sparse)


class PopulationConfig(Parameterized):
    n_agents = param.Integer(20)
    box_size = param.Number(100., bounds=(0, None))

    @param.depends('n_agents', 'box_size', watch=True, on_init=True)
    def _update_population_arrays(self):
        print('upd population')
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        self.positions = self.box_size * jax.random.uniform(subkey, (self.n_agents, 2))
        key, subkey = jax.random.split(key)
        self.thetas = jax.random.uniform(subkey, (self.n_agents,), maxval=2 * np.pi)

def behavior_test_1(proxs):
    return proxs

def behavior_test_2(proxs):
    return jnp.array([1., 0.])


class BehaviorConfig(Parameterized):
    behavior_bank = Parameter([behavior_test_1, behavior_test_2])
    entity_behaviors = Parameter(jnp.hstack((jnp.zeros(10, dtype=int), jnp.ones(10, dtype=int))))