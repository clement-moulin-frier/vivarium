import param
from param import Parameterized
import numpy as np

import vivarium.simulator.behaviors as behaviors
from vivarium.simulator.sim_computation import EntityType

from jax_md.rigid_body import monomer


mass = monomer.mass()
mass_center = float(mass.center[0])
mass_orientation = float(mass.orientation[0])

class Config(Parameterized):
    export_fields_include = param.List(None, allow_None=True)
    export_fields_exclude = param.List(None, allow_None=True)
    export_fields = param.List(None, allow_None=True)


    @param.depends('export_fields_include', 'export_fields_exclude', watch=True, on_init=True)
    def _update_fields(self):
        if self.export_fields_include is None:
            d = self.param.values()
            del d['name']
            self.export_fields_include = list(d.keys())
        if self.export_fields_exclude is None:
            self.export_fields_exclude = []
        self.export_fields_exclude += ['export_fields_include', 'export_fields_exclude', 'export_fields']
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
    idx = param.Integer()
    x_position = param.Number(0.)
    y_position = param.Number(0.)
    orientation = param.Number(0.)
    mass_center = param.Number(mass_center)
    mass_orientation = param.Number(mass_orientation)
    behavior = param.ObjectSelector(default=behaviors.linear_behavior_enum.AGGRESSION.name,
                                    objects=behaviors.behavior_name_map.keys())
    left_motor = param.Number(0., bounds=(0., 1.))
    right_motor = param.Number(0., bounds=(0., 1.))
    left_prox = param.Number(0., bounds=(0., 1.))
    right_prox = param.Number(0., bounds=(0., 1.))
    wheel_diameter = param.Number(2.)
    diameter = param.Number(5.)
    speed_mul = param.Number(0.1)
    theta_mul = param.Number(0.1)
    proxs_dist_max = param.Number(100., bounds=(0, None))
    proxs_cos_min = param.Number(0., bounds=(-1., 1.))
    color = param.Color('blue')
    friction = param.Number(1e-1)

    def __init__(self, **params):
        super().__init__(**params)


class ObjectConfig(Config):
    idx = param.Integer()
    x_position = param.Number(0.)
    y_position = param.Number(0.)
    orientation = param.Number(0.)
    mass_center = param.Number(mass_center)
    mass_orientation = param.Number(mass_orientation)
    diameter = param.Number(5.)
    color = param.Color('red')
    friction = param.Number(10.)

    def __init__(self, **params):
        super().__init__(**params)


config_to_etype = {AgentConfig: EntityType.AGENT, ObjectConfig: EntityType.OBJECT}


class SimulatorConfig(Config):
    box_size = param.Number(100., bounds=(0, None))
    map_dim = param.Integer(2, bounds=(1, None))
    n_agents = param.Integer(10)
    n_objects = param.Integer(2)
    num_steps_lax = param.Integer(4)
    dt = param.Number(0.1)
    freq = param.Number(40., allow_None=True)
    neighbor_radius = param.Number(100., bounds=(0, None))
    to_jit = param.Boolean(True)
    use_fori_loop = param.Boolean(False)
    dynamics_fn = param.Parameter()
    behavior_bank = param.List(behaviors.behavior_bank)

    def __init__(self, **params):
        super().__init__(**params)
        self.export_fields_exclude = ['dynamics_fn', 'behavior_bank']

