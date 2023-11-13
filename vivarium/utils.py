import jax
import jax.numpy as jnp
import numpy as np

import jax_md
from jax_md.util import f32
from jax_md.rigid_body import RigidBody

import dataclasses
import typing

from vivarium.simulator.config import AgentConfig, ObjectConfig, etype_to_config
from vivarium.simulator.sim_computation import State, NVEState, AgentState, ObjectState, EntityType
from vivarium.simulator.behaviors import behavior_name_map, reversed_behavior_name_map

import matplotlib.colors as mcolors


agent_config_fields = AgentConfig.param.objects().keys()
agent_state_fields = [f.name for f in jax_md.dataclasses.fields(AgentState)]

agent_common_fields = [f for f in agent_config_fields if f in agent_state_fields]

object_config_fields = ObjectConfig.param.objects().keys()
object_state_fields = [f.name for f in jax_md.dataclasses.fields(ObjectState)]

object_common_fields = [f for f in object_config_fields if f in object_state_fields]

state_fields_dict = {EntityType.AGENT: agent_state_fields,
                     EntityType.OBJECT: object_state_fields}

@dataclasses.dataclass
class StateFieldInfo:
    nested_field: typing.Tuple
    column_idx: np.array
    state_to_config: typing.Callable
    config_to_state: typing.Callable


identity_s_to_c = lambda x, typ: typ(x)
identity_c_to_s = lambda x: x
behavior_s_to_c = lambda x, typ: reversed_behavior_name_map[int(x)]
behavior_c_to_s = lambda x: behavior_name_map[x]
color_s_to_c = lambda x, typ: mcolors.to_hex(x)  # Warning : temporary (below as well)
color_c_to_s = lambda x: mcolors.to_rgb(x)
mass_center_s_to_c = lambda x, typ: typ(x)
mass_center_c_to_s = lambda x: [x]


agent_configs_to_state_dict = {'x_position': StateFieldInfo(('nve_state', 'position', 'center'), np.array([0]), identity_s_to_c, identity_c_to_s),
                               'y_position': StateFieldInfo(('nve_state', 'position', 'center'), np.array([1]), identity_s_to_c, identity_c_to_s),
                               'orientation': StateFieldInfo(('nve_state', 'position', 'orientation'), None, identity_s_to_c, identity_c_to_s),
                               'mass_center': StateFieldInfo(('nve_state', 'mass', 'center'), np.array([0]), mass_center_s_to_c, mass_center_c_to_s),
                               'mass_orientation': StateFieldInfo(('nve_state', 'mass', 'orientation'), None, identity_s_to_c, identity_c_to_s),
                               'diameter': StateFieldInfo(('nve_state', 'diameter'), None, identity_s_to_c, identity_c_to_s),
                               'friction': StateFieldInfo(('nve_state', 'friction'), None, identity_s_to_c, identity_c_to_s),
                               'left_motor': StateFieldInfo(('agent_state', 'motor',), np.array([0]), identity_s_to_c, identity_c_to_s),
                               'right_motor': StateFieldInfo(('agent_state', 'motor',), np.array([1]), identity_s_to_c, identity_c_to_s),
                               'left_prox': StateFieldInfo(('agent_state', 'prox',), np.array([0]), identity_s_to_c, identity_c_to_s),
                               'right_prox': StateFieldInfo(('agent_state', 'prox',), np.array([1]), identity_s_to_c, identity_c_to_s),
                               'behavior': StateFieldInfo(('agent_state', 'behavior',), None, behavior_s_to_c, behavior_c_to_s),
                               'color': StateFieldInfo(('agent_state', 'color',), np.arange(3), color_s_to_c, color_c_to_s),
                               'idx': StateFieldInfo(('agent_state', 'nve_idx',), None, identity_s_to_c, identity_c_to_s)
                               }

agent_configs_to_state_dict.update({f: StateFieldInfo(('agent_state', f,), None, identity_s_to_c, identity_c_to_s) for f in agent_common_fields if f not in agent_configs_to_state_dict})

object_configs_to_state_dict = {'x_position': StateFieldInfo(('nve_state', 'position', 'center'), np.array([0]), identity_s_to_c, identity_c_to_s),
                               'y_position': StateFieldInfo(('nve_state', 'position', 'center'), np.array([1]), identity_s_to_c, identity_c_to_s),
                               'orientation': StateFieldInfo(('nve_state', 'position', 'orientation'), None, identity_s_to_c, identity_c_to_s),
                               'mass_center': StateFieldInfo(('nve_state', 'mass', 'center'), np.array([0]), mass_center_s_to_c, mass_center_c_to_s),
                               'mass_orientation': StateFieldInfo(('nve_state', 'mass', 'orientation'), None, identity_s_to_c, identity_c_to_s),
                               'diameter': StateFieldInfo(('nve_state', 'diameter'), None, identity_s_to_c, identity_c_to_s),
                               'friction': StateFieldInfo(('nve_state', 'friction'), None, identity_s_to_c, identity_c_to_s),
                               'color': StateFieldInfo(('object_state', 'color',), np.arange(3), color_s_to_c, color_c_to_s),
                               'idx': StateFieldInfo(('object_state', 'nve_idx',), None, identity_s_to_c, identity_c_to_s)

                               }

object_configs_to_state_dict.update({f: StateFieldInfo(('object_state', f,), None, identity_s_to_c, identity_c_to_s) for f in object_common_fields if f not in object_configs_to_state_dict})

configs_to_state_dict = {EntityType.AGENT: agent_configs_to_state_dict,
                         EntityType.OBJECT: object_configs_to_state_dict}


def get_default_state(n_entities_dict):
    n_agents = n_entities_dict[EntityType.AGENT]
    n_objects = n_entities_dict[EntityType.OBJECT]
    return State(nve_state=NVEState(position=RigidBody(center=jnp.zeros((n_agents + n_objects, 2)), orientation=jnp.zeros(n_agents + n_objects)),
                                    momentum=None,
                                    force=RigidBody(center=jnp.zeros((n_agents + n_objects, 2)), orientation=jnp.zeros(n_agents + n_objects)),
                                    mass=RigidBody(center=jnp.zeros((n_agents + n_objects, 1)), orientation=jnp.zeros(n_agents + n_objects)),
                                    entity_type=jnp.array([EntityType.AGENT.value] * n_agents + [EntityType.OBJECT.value] * n_objects, dtype=int),
                                    entity_idx = jnp.array(list(range(n_agents)) + list(range(n_objects))),
                                    diameter=jnp.zeros(n_agents + n_objects),
                                    friction=jnp.zeros(n_agents + n_objects)
                                    ),
                 agent_state=AgentState(nve_idx=jnp.zeros(n_agents, dtype=int),
                                        prox=jnp.zeros((n_agents, 2)),
                                        motor=jnp.zeros((n_agents, 2)),
                                        behavior=jnp.zeros(n_agents, dtype=int),
                                        wheel_diameter=jnp.zeros(n_agents),
                                        speed_mul=jnp.zeros(n_agents),
                                        theta_mul=jnp.zeros(n_agents),
                                        proxs_dist_max=jnp.zeros(n_agents),
                                        proxs_cos_min=jnp.zeros(n_agents),
                                        color=jnp.zeros((n_agents, 3))),
                 object_state=ObjectState(nve_idx=jnp.zeros(n_objects, dtype=int), color=jnp.zeros((n_objects, 3))))




def rec_set_dataclass(var, nested_field, row_idx, column_idx, value):

    assert len(nested_field) > 0

    if len(nested_field) == 1:
        field = nested_field[0]
        if column_idx is None or len(column_idx) == 0:
            d = {field: getattr(var, field).at[row_idx].set(value.reshape(-1))}
        else:
            d = {field: getattr(var, field).at[jnp.ix_(row_idx, column_idx)].set(value.reshape(len(row_idx), len(column_idx)))}
        return d
    else:
        next_var = getattr(var, nested_field[0])
        d = rec_set_dataclass(next_var, nested_field[1:], row_idx, column_idx, value)
        return {nested_field[0]: next_var.set(**d)}


def set_state_from_config_dict(config_dict, state=None):
    n_entities_dict = {etype: len(config) for etype, config in config_dict.items()}
    state = state or get_default_state(n_entities_dict)
    e_idx = jnp.zeros(sum(n_entities_dict.values()), dtype=int)
    for e_type, configs in config_dict.items():
        params = configs[0].param_names()
        for p in params:
            state_field_info = configs_to_state_dict[e_type][p]
            nve_idx = [c.idx for c in configs] if state_field_info.nested_field[0] == 'nve_state' else range(len(configs))
            change = rec_set_dataclass(state, state_field_info.nested_field, jnp.array(nve_idx), state_field_info.column_idx,
                                       jnp.array([state_field_info.config_to_state(getattr(c, p)) for c in configs]))
            state = state.set(**change)
        e_idx.at[getattr(state, f'{e_type.name.lower()}_state').nve_idx].set(jnp.array(range(n_entities_dict[e_type])))
    change = rec_set_dataclass(state, ('nve_state', 'entity_idx'), jnp.array(range(sum(n_entities_dict.values()))), None, e_idx)
    state.set(**change)
    return state


def set_state_from_agent_configs(agent_configs, state=None, params=None):
    state = state or get_default_state(len(agent_configs))
    params = params or agent_configs[0].param_names()
    for p in params:
        state_field_info = agent_configs_to_state_dict[p]
        agent_idx = [a.idx for a in agent_configs] if state_field_info.nested_field[0] == 'nve_state' else range(len(agent_configs))
        change = rec_set_dataclass(state, state_field_info.nested_field, jnp.array(agent_idx), state_field_info.column_idx,
                                   jnp.array([state_field_info.config_to_state(getattr(c, p)) for c in agent_configs]))
        state = state.set(**change)
    return state


def set_configs_from_state(state, config_dict=None):
    if config_dict is None:
        config_dict = {etype: [] for etype in list(EntityType)}
        for idx, etype_int in enumerate(state.nve_state.entity_type):
            etype = EntityType(etype_int)
            config_dict[etype].append(etype_to_config[etype](idx=idx))
    for e_type in list(EntityType):  # [EntityType.AGENT, EntityType.OBJECT]:
        for param, state_field_info in configs_to_state_dict[e_type].items():
            value = state
            for f in state_field_info.nested_field:
                value = getattr(value, f)
            for config in config_dict[e_type]:
                t = type(getattr(config, param))
                # value = np.array(value).astype(t)
                  # TODO: use state.row_idx(.)
                row_idx = config.idx if state_field_info.nested_field[0] == 'nve_state' else state.nve_state.entity_idx[config.idx]
                if state_field_info.column_idx is None:
                    value_to_set = value[row_idx]
                else:
                    value_to_set = value[row_idx, state_field_info.column_idx]
                value_to_set = state_field_info.state_to_config(value_to_set, t)
                config.param.update(**{param: value_to_set})
    return config_dict


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


def configs_to_array_dict(config_dict, fields=None):
    state = set_state_from_config_dict(config_dict)
    if fields is None:
        fields = []
        for e_type, configs in config_dict.items():
            fields.extend(state_fields_dict[e_type])
    set_state_from_agent_configs(con)
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


######## LEGACY code ######

def config_attribute_as_array(configs, attr):
    dtype = type(getattr(configs[0], attr))
    return jnp.array([f32(getattr(config, attr)) for config in configs], dtype=dtype)