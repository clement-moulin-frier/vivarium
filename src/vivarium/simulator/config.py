import param
from param import Parameterized, Parameter
from jax_md import space, partition
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

import vivarium.simulator.behaviors as behaviors
from vivarium.simulator.sim_computation import Population


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
    #box_size = param.Number(100., bounds=(0, None))

    #@param.depends('n_agents', 'box_size', watch=True, on_init=True)
    def generate_population(self, box_size):
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        positions = box_size * jax.random.uniform(subkey, (self.n_agents, 2))
        key, subkey = jax.random.split(key)
        thetas = jax.random.uniform(subkey, (self.n_agents,), maxval=2 * np.pi)
        proxs = jnp.zeros((self.n_agents, 2))
        motors = jnp.zeros((self.n_agents, 2))
        return Population(positions=positions, thetas=thetas, proxs=proxs, motors=motors, entity_type=0)

    def json(self):
        return self.param.serialize_parameters()


class BehaviorConfig(Parameterized):
    population_config = param.ClassSelector(PopulationConfig, instantiate=False)
    behavior_bank = param.List([partial(behaviors.linear_behavior,
                                               matrix=behaviors.linear_behavior_matrices[beh])
                                       for beh in behaviors.linear_behavior_enum] + [behaviors.apply_motors])
    behavior_name_map = {beh.name: i for i, beh in enumerate(behaviors.linear_behavior_enum)}
                                       #for beh in behaviors.linear_behavior_enum}.update({'manual': behaviors.apply_motors}))

    entity_behaviors = param.ClassSelector(jax.Array)

    @param.depends('population_config.n_agents', watch=True, on_init=True)
    def _update_behaviors(self):
        self.behavior_name_map['manual'] = len(self.behavior_bank) - 1
        #self.behavior_bank = self.predefined_behaviors + [behaviors.noop] * self.population_config.n_agents
        self.entity_behaviors = jnp.zeros(self.population_config.n_agents, dtype=int)
