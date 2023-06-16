import jax
import jax.numpy as jnp
import numpy as np

import jax_md
from jax_md.util import f32
from jax_md.rigid_body import RigidBody

import dataclasses
import typing

from vivarium.simulator.config import AgentConfig
from vivarium.simulator.sim_computation import NVEState
from vivarium.simulator.behaviors import behavior_name_map, reversed_behavior_name_map

import matplotlib.colors as mcolors


config_fields = AgentConfig.param.objects().keys()
state_fields = [f.name for f in jax_md.dataclasses.fields(NVEState)]

common_fields = [f for f in config_fields if f in state_fields]


@dataclasses.dataclass
class StateFieldInfo:
    nested_field: typing.Tuple
    column_idx: int
    state_to_config: typing.Callable
    config_to_state: typing.Callable


identity_s_to_c = lambda x, typ: typ(x)
identity_c_to_s = lambda x: x
behavior_s_to_c = lambda x, typ: reversed_behavior_name_map[int(x)]
behavior_c_to_s = lambda x: behavior_name_map[x]
color_s_to_c = lambda x, typ: mcolors.to_hex(x)  # Warning : temporary (below as well)
color_c_to_s = lambda x: mcolors.to_rgb(x)


agent_configs_to_state_dict = {'x_position': StateFieldInfo(('position', 'center'), 0, identity_s_to_c, identity_c_to_s),
                               'y_position': StateFieldInfo(('position', 'center'), 1, identity_s_to_c, identity_c_to_s),
                               'orientation': StateFieldInfo(('position', 'orientation'), None, identity_s_to_c, identity_c_to_s),
                               'mass_center': StateFieldInfo(('mass', 'center'), None, identity_s_to_c, identity_c_to_s),
                               'mass_orientation': StateFieldInfo(('mass', 'orientation'), None, identity_s_to_c, identity_c_to_s),
                               'left_motor': StateFieldInfo(('motor',), 0, identity_s_to_c, identity_c_to_s),
                               'right_motor': StateFieldInfo(('motor',), 1, identity_s_to_c, identity_c_to_s),
                               'left_prox': StateFieldInfo(('prox',), 0, identity_s_to_c, identity_c_to_s),
                               'right_prox': StateFieldInfo(('prox',), 1, identity_s_to_c, identity_c_to_s),
                               'behavior': StateFieldInfo(('behavior',), None, behavior_s_to_c, behavior_c_to_s),
                               'color': StateFieldInfo(('color',), None, color_s_to_c, color_c_to_s)
                               }

agent_configs_to_state_dict.update({f: StateFieldInfo((f,), None, identity_s_to_c, identity_c_to_s) for f in common_fields if f not in agent_configs_to_state_dict})


def get_default_state(n_agents):
    return NVEState(position=RigidBody(center=jnp.zeros((n_agents, 2)), orientation=jnp.zeros(n_agents)),
                    momentum=RigidBody(center=jnp.zeros((n_agents, 2)), orientation=jnp.zeros(n_agents)),
                    force=RigidBody(center=jnp.zeros((n_agents, 2)), orientation=jnp.zeros(n_agents)),
                    mass=RigidBody(center=jnp.zeros(n_agents), orientation=jnp.zeros(n_agents)),
                    prox=jnp.zeros((n_agents, 2)),
                    motor=jnp.zeros((n_agents, 2)),
                    behavior=jnp.zeros(n_agents, dtype=int),
                    wheel_diameter=jnp.zeros(n_agents),
                    base_length=jnp.zeros(n_agents),
                    speed_mul=jnp.zeros(n_agents),
                    theta_mul=jnp.zeros(n_agents),
                    proxs_dist_max=jnp.zeros(n_agents),
                    proxs_cos_min=jnp.zeros(n_agents),
                    color=jnp.zeros((n_agents, 3)),
                    entity_type=jnp.zeros(n_agents, dtype=int),
                    idx=jnp.zeros(n_agents, dtype=int)
                    )


def config_attribute_as_array(agent_configs, attr):
    dtype = type(getattr(agent_configs[0], attr))
    return jnp.array([f32(getattr(config, attr)) for config in agent_configs], dtype=dtype)


def rec_set_dataclass(var, nested_field, row_idx, column_idx, value):

    assert len(nested_field) > 0

    if len(nested_field) == 1:
        field = nested_field[0]
        if column_idx is None:
            d = {field: getattr(var, field).at[row_idx].set(value)}
        else:
            d = {field: getattr(var, field).at[row_idx, column_idx].set(value)}
        return d
    else:
        next_var = getattr(var, nested_field[0])
        d = rec_set_dataclass(next_var, nested_field[1:], row_idx, column_idx, value)
        return {nested_field[0]: next_var.set(**d)}


def set_state_from_agent_configs(agent_configs, state=None, params=None):
    state = state or get_default_state(len(agent_configs))
    agent_idx = [a.idx for a in agent_configs]
    params = params or agent_configs[0].to_dict().keys()
    for p in params:
        state_field_info = agent_configs_to_state_dict[p]
        change = rec_set_dataclass(state, state_field_info.nested_field, jnp.array(agent_idx), state_field_info.column_idx,
                                   jnp.array([state_field_info.config_to_state(getattr(c, p)) for c in agent_configs]))
        state = state.set(**change)
    return state


def set_agent_configs_from_state(state, agent_configs, first_nested_fields=['position', 'prox', 'motor', 'behavior',
                                                                               'wheel_diameter', 'base_length',
                                                                               'speed_mul', 'theta_mul',
                                                                               'proxs_dist_max', 'proxs_cos_min',
                                                                               'entity_type']):
    for field in first_nested_fields:
        for param, state_field_info in agent_configs_to_state_dict.items():
            if field == state_field_info.nested_field[0]:
                value = state
                for f in state_field_info.nested_field:
                    value = getattr(value, f)
                for config in agent_configs:
                    t = type(getattr(config, param))
                    # value = np.array(value).astype(t)
                    if state_field_info.column_idx is None:
                        value_to_set = value[config.idx]
                    else:
                        value_to_set = value[config.idx, state_field_info.column_idx]
                    value_to_set = state_field_info.state_to_config(value_to_set, t)
                    config.param.update(**{param: value_to_set})


def agent_configs_to_array_dict(agent_configs, fields=None):
    fields = fields or state_fields
    state = set_state_from_agent_configs(agent_configs)
    return {f: getattr(state, f) for f in fields}


# def generate_positions_orientations(key, n_agents, box_size):
#     key, subkey = jax.random.split(key)
#     positions = box_size * jax.random.uniform(subkey, (n_agents, 2))
#     key, subkey = jax.random.split(key)
#     orientations = jax.random.uniform(subkey, (n_agents,), maxval=2 * np.pi)
#     return key, positions, orientations

def get_init_state_kwargs(agent_configs):
    # key, subkey = jax.random.split(key)
    # key, positions, orientations = generate_positions_orientations(key=key,
    #                                                           n_agents=len(agent_configs),
    #                                                           box_size=box_size)
    #
    # rigid_body = self.agent_configs_as_array_dict(fields=['x_position', 'y_position', 'orientation'])['position']
    state_kwargs = agent_configs_to_array_dict(agent_configs, fields=['idx', 'position', 'mass',
                                                                                 'prox', 'motor', 'behavior',
                                                                                 'wheel_diameter',
                                                                                 'base_length',
                                                                                 'speed_mul',
                                                                                 'theta_mul',
                                                                                 'proxs_dist_max',
                                                                                 'proxs_cos_min',
                                                                                 'color',
                                                                                 'entity_type'
                                                                            ])
    return state_kwargs