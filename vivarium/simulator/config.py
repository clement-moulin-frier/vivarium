import param
from param import Parameterized

import vivarium.simulator.behaviors as behaviors

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
    behavior = param.ObjectSelector(default=behaviors.linear_behavior_enum.AGGRESSION.name,
                                    objects=behaviors.behavior_name_map.keys())
    wheel_diameter = param.Number(2.)
    base_length = param.Number(10.)
    speed_mul = param.Number(0.1)
    theta_mul = param.Number(0.1)
    proxs_dist_max = param.Number(100., bounds=(0, None))
    proxs_cos_min = param.Number(0., bounds=(-1., 1.))
    entity_type = param.Integer(0)


class SimulatorConfig(Config):
    box_size = param.Number(100., bounds=(0, None))
    map_dim = param.Integer(2, bounds=(1, None))
    n_agents = param.Integer(30)
    num_steps_lax = param.Integer(50)
    num_lax_loops = param.Integer(1)
    dt = param.Number(0.1)
    freq = param.Number(40., allow_None=True)
    neighbor_radius = param.Number(100., bounds=(0, None))
    to_jit = param.Boolean(True)
    use_fori_loop = param.Boolean(False)
