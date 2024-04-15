import param
from param import Parameterized

import vivarium.simulator.behaviors as behaviors
from vivarium.simulator.states import StateType

from jax_md.rigid_body import monomer

import numpy as np


mass = monomer.mass()
mass_center = float(mass.center[0])
mass_orientation = float(mass.orientation[0])


class Config(Parameterized):

    def to_dict(self, params=None):
        d = self.param.values()
        del d['name']
        if params is not None:
            return {p: d[p] for p in params}
        else:
            return d

    def param_names(self):
        return list(self.to_dict().keys())

    def json(self):
        return self.param.serialize_parameters(subset=self.param_names())


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
    neighbor_map_dist = param.Array(np.array([0.]))
    neighbor_map_theta = param.Array(np.array([0.]))
    wheel_diameter = param.Number(2.)
    diameter = param.Number(5.)
    speed_mul = param.Number(1.)
    max_speed = param.Number(10.)
    theta_mul = param.Number(1.)
    proxs_dist_max = param.Number(100., bounds=(0, None))
    proxs_cos_min = param.Number(0., bounds=(-1., 1.))
    color = param.Color('blue')
    friction = param.Number(1e-1)
    exists = param.Boolean(True)

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
    friction = param.Number(0.1)
    exists = param.Boolean(True)

    def __init__(self, **params):
        super().__init__(**params)


class SimulatorConfig(Config):
    idx = param.Integer(0, constant=True)
    box_size = param.Number(100., bounds=(0, None))
    max_agents = param.Integer(10)
    max_objects = param.Integer(2)
    num_steps_lax = param.Integer(4)
    dt = param.Number(0.1)
    freq = param.Number(40., allow_None=True)
    neighbor_radius = param.Number(100., bounds=(0, None))
    to_jit = param.Boolean(True)
    use_fori_loop = param.Boolean(False)
    collision_eps = param.Number(0.1)
    collision_alpha = param.Number(0.5)

    def __init__(self, **params):
        super().__init__(**params)


config_to_stype = {SimulatorConfig: StateType.SIMULATOR, AgentConfig: StateType.AGENT, ObjectConfig: StateType.OBJECT}
stype_to_config = {stype: config_class for config_class, stype in config_to_stype.items()}