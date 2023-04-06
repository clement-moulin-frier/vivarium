import param
from param import Parameterized, Parameter
from jax_md import space, partition
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

import vivarium.simulator.behaviors as behaviors
from vivarium.simulator.sim_computation import Population

class Config(Parameterized):
    export_fields_include = param.List(None, allow_None=True)
    export_fields_exclude = param.List(None, allow_None=True)

    @param.depends('export_fields_include', 'export_fields_exclude', watch=True, on_init=True)
    def _update_fields(self):
        if self.export_fields_include is None:
            d = self.param.values()
            del d['name']
            self.export_fields_include = list(d.keys())
        if self.export_fields_exclude is None:
            self.export_fields_exclude = []
        self.export_fields_exclude += ['export_fields_include', 'export_fields_exclude']
        self.export_fields = [f for f in self.export_fields_include if f not in self.export_fields_exclude]

    def to_dict(self, fields=None):
        d = self.param.values()
        if fields is None:
            return {f: d[f] for f in self.export_fields}
        else:
            return {f: d[f] for f in fields}

    def json(self):
        return self.param.serialize_parameters(subset=self.export_fields)

class AgentConfig(Config):

    wheel_diameter = param.Number(2.)
    base_length = param.Number(10.)
    speed_mul = param.Number(0.1)
    theta_mul = param.Number(0.1)
    neighbor_radius = param.Number(100., bounds=(0, None))
    proxs_dist_max = param.Number(100., bounds=(0, None))
    proxs_cos_min = param.Number(0., bounds=(-1., 1.))


class SimulatorConfig(Config):
    box_size = param.Number(100., bounds=(0, None))
    map_dim = param.Integer(2, bounds=(1, None))
    num_steps_lax = param.Integer(50)
    num_lax_loops = param.Integer(1)
    dt = param.Number(0.1)
    freq = param.Number(100.)
    to_jit = param.Boolean(True)


class PopulationConfig(Config):
    n_agents = param.Integer(20)

class EngineConfig(param.Parameterized):
    displacement = param.Parameter()
    shift = param.Parameter()
    behavior_bank = param.List([partial(behaviors.linear_behavior,
                                        matrix=behaviors.linear_behavior_matrices[beh])
                                for beh in behaviors.linear_behavior_enum] + [behaviors.apply_motors])
    behavior_name_map = {beh.name: i for i, beh in enumerate(behaviors.linear_behavior_enum)}
                                       #for beh in behaviors.linear_behavior_enum}.update({'manual': behaviors.apply_motors}))

    entity_behaviors = param.Array()

# class BehaviorConfig(Config):
#     population_config = param.ClassSelector(PopulationConfig, instantiate=False)
#     behavior_bank = param.List([partial(behaviors.linear_behavior,
#                                                matrix=behaviors.linear_behavior_matrices[beh])
#                                        for beh in behaviors.linear_behavior_enum] + [behaviors.apply_motors])
#     behavior_name_map = {beh.name: i for i, beh in enumerate(behaviors.linear_behavior_enum)}
#                                        #for beh in behaviors.linear_behavior_enum}.update({'manual': behaviors.apply_motors}))
#
#     entity_behaviors = param.ClassSelector(jax.Array)
#
#     export_fields_include = param.List(['behavior_name_map', 'entity_behaviors'])
#     export_fields_exclude = param.List(['population_config'])

