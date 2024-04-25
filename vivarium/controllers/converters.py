import typing
import dataclasses
from collections import namedtuple, defaultdict

import jax_md
import jax.numpy as jnp
import numpy as np
import matplotlib.colors as mcolors

from jax_md.rigid_body import RigidBody

from vivarium.controllers.config import AgentConfig, ObjectConfig, SimulatorConfig, stype_to_config, config_to_stype
from vivarium.simulator.states import State, SimulatorState, EntityState, AgentState, ObjectState, EntityType, StateType
from vivarium.simulator.behaviors import behavior_name_map, reversed_behavior_name_map


agent_config_fields = AgentConfig.param.objects().keys()
agent_state_fields = [f.name for f in jax_md.dataclasses.fields(AgentState)]

agent_common_fields = [f for f in agent_config_fields if f in agent_state_fields]

object_config_fields = ObjectConfig.param.objects().keys()
object_state_fields = [f.name for f in jax_md.dataclasses.fields(ObjectState)]

object_common_fields = [f for f in object_config_fields if f in object_state_fields]

simulator_config_fields = SimulatorConfig.param.objects().keys()
simulator_state_fields = [f.name for f in jax_md.dataclasses.fields(SimulatorState)]

simulator_common_fields = [f for f in simulator_config_fields if f in simulator_state_fields]
#
state_fields_dict = {StateType.AGENT: agent_state_fields,
                     StateType.OBJECT: object_state_fields,
                     StateType.SIMULATOR: simulator_state_fields}


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
color_s_to_c = lambda x, typ: mcolors.to_hex(np.array(x))  # Warning : temporary (below as well)
color_c_to_s = lambda x: mcolors.to_rgb(x)
mass_center_s_to_c = lambda x, typ: typ(x)
mass_center_c_to_s = lambda x: [x]
exists_c_to_s = lambda x: int(x)


agent_configs_to_state_dict = {'x_position': StateFieldInfo(('entity_state', 'position', 'center'), 0, identity_s_to_c, identity_c_to_s),
                               'y_position': StateFieldInfo(('entity_state', 'position', 'center'), 1, identity_s_to_c, identity_c_to_s),
                               'orientation': StateFieldInfo(('entity_state', 'position', 'orientation'), None, identity_s_to_c, identity_c_to_s),
                               'mass_center': StateFieldInfo(('entity_state', 'mass', 'center'), np.array([0]), mass_center_s_to_c, mass_center_c_to_s),
                               'mass_orientation': StateFieldInfo(('entity_state', 'mass', 'orientation'), None, identity_s_to_c, identity_c_to_s),
                               'diameter': StateFieldInfo(('entity_state', 'diameter'), None, identity_s_to_c, identity_c_to_s),
                               'friction': StateFieldInfo(('entity_state', 'friction'), None, identity_s_to_c, identity_c_to_s),
                               'left_motor': StateFieldInfo(('agent_state', 'motor',), 0, identity_s_to_c, identity_c_to_s),
                               'right_motor': StateFieldInfo(('agent_state', 'motor',), 1, identity_s_to_c, identity_c_to_s),
                               'left_prox': StateFieldInfo(('agent_state', 'prox',), 0, identity_s_to_c, identity_c_to_s),
                               'right_prox': StateFieldInfo(('agent_state', 'prox',), 1, identity_s_to_c, identity_c_to_s),
                               'behavior': StateFieldInfo(('agent_state', 'behavior',), None, behavior_s_to_c, behavior_c_to_s),
                               'color': StateFieldInfo(('agent_state', 'color',), np.arange(3), color_s_to_c, color_c_to_s),
                               'idx': StateFieldInfo(('agent_state', 'nve_idx',), None, identity_s_to_c, identity_c_to_s),
                               'exists': StateFieldInfo(('entity_state', 'exists'), None, identity_s_to_c, exists_c_to_s)
                               }

agent_configs_to_state_dict.update({f: StateFieldInfo(('agent_state', f,), None, identity_s_to_c, identity_c_to_s) for f in agent_common_fields if f not in agent_configs_to_state_dict})

object_configs_to_state_dict = {'x_position': StateFieldInfo(('entity_state', 'position', 'center'), 0, identity_s_to_c, identity_c_to_s),
                                'y_position': StateFieldInfo(('entity_state', 'position', 'center'), 1, identity_s_to_c, identity_c_to_s),
                                'orientation': StateFieldInfo(('entity_state', 'position', 'orientation'), None, identity_s_to_c, identity_c_to_s),
                                'mass_center': StateFieldInfo(('entity_state', 'mass', 'center'), np.array([0]), mass_center_s_to_c, mass_center_c_to_s),
                                'mass_orientation': StateFieldInfo(('entity_state', 'mass', 'orientation'), None, identity_s_to_c, identity_c_to_s),
                                'diameter': StateFieldInfo(('entity_state', 'diameter'), None, identity_s_to_c, identity_c_to_s),
                                'friction': StateFieldInfo(('entity_state', 'friction'), None, identity_s_to_c, identity_c_to_s),
                                'color': StateFieldInfo(('object_state', 'color',), np.arange(3), color_s_to_c, color_c_to_s),
                                'idx': StateFieldInfo(('object_state', 'nve_idx',), None, identity_s_to_c, identity_c_to_s),
                                'exists': StateFieldInfo(('entity_state', 'exists'), None, identity_s_to_c, exists_c_to_s)

                                }

object_configs_to_state_dict.update({f: StateFieldInfo(('object_state', f,), None, identity_s_to_c, identity_c_to_s) for f in object_common_fields if f not in object_configs_to_state_dict})

simulator_configs_to_state_dict = {}
simulator_configs_to_state_dict.update({f: StateFieldInfo(('simulator_state', f,), None, identity_s_to_c, identity_c_to_s) for f in simulator_common_fields if f not in simulator_configs_to_state_dict})

configs_to_state_dict = {StateType.AGENT: agent_configs_to_state_dict,
                         StateType.OBJECT: object_configs_to_state_dict,
                         StateType.SIMULATOR: simulator_configs_to_state_dict
                         }


def get_default_state(n_entities_dict):
    max_agents = n_entities_dict[StateType.AGENT]
    max_objects = n_entities_dict[StateType.OBJECT]
    n_entities = sum(n_entities_dict.values())
    return State(simulator_state=SimulatorState(idx=jnp.array([0]), box_size=jnp.array([100.]),
                                                max_agents=jnp.array([max_agents]), max_objects=jnp.array([max_objects]),
                                                num_steps_lax=jnp.array([1]), dt=jnp.array([1.]), freq=jnp.array([1.]),
                                                neighbor_radius=jnp.array([1.]),
                                                to_jit= jnp.array([1]), use_fori_loop=jnp.array([0]),
                                                collision_alpha=jnp.array([0.]),
                                                collision_eps=jnp.array([0.])),
                 entity_state=EntityState(position=RigidBody(center=jnp.zeros((n_entities, 2)), orientation=jnp.zeros(n_entities)),
                                    momentum=None,
                                    force=RigidBody(center=jnp.zeros((n_entities, 2)), orientation=jnp.zeros(n_entities)),
                                    mass=RigidBody(center=jnp.zeros((n_entities, 1)), orientation=jnp.zeros(n_entities)),
                                    entity_type=jnp.array([EntityType.AGENT.value] * max_agents + [EntityType.OBJECT.value] * max_objects, dtype=int),
                                    entity_idx = jnp.array(list(range(max_agents)) + list(range(max_objects))),
                                    diameter=jnp.zeros(n_entities),
                                    friction=jnp.zeros(n_entities),
                                    exists=jnp.ones(n_entities, dtype=int)
                                    ),
                 agent_state=AgentState(nve_idx=jnp.zeros(max_agents, dtype=int),
                                        prox=jnp.zeros((max_agents, 2)),
                                        motor=jnp.zeros((max_agents, 2)),
                                        behavior=jnp.zeros(max_agents, dtype=int),
                                        wheel_diameter=jnp.zeros(max_agents),
                                        speed_mul=jnp.zeros(max_agents),
                                        max_speed=jnp.zeros(max_agents),
                                        theta_mul=jnp.zeros(max_agents),
                                        proxs_dist_max=jnp.zeros(max_agents),
                                        proxs_cos_min=jnp.zeros(max_agents),
                                        color=jnp.zeros((max_agents, 3))),
                 object_state=ObjectState(nve_idx=jnp.zeros(max_objects, dtype=int), color=jnp.zeros((max_objects, 3))))


EntityTuple = namedtuple('EntityTuple', ['idx', 'col', 'val'])
ValueTuple = namedtuple('ValueData', ['nve_idx', 'col_idx', 'row_map', 'col_map', 'val'])
StateChangeTuple = namedtuple('StateChange', ['nested_field', 'nve_idx', 'column_idx', 'value'])


def events_to_nve_data(events, state):
    nve_data = defaultdict(list)
    for e in events:
        config = e.obj
        param = e.name
        etype = config_to_stype[config.param.cls]
        state_field_info = configs_to_state_dict[etype][param]
        nested_field = state_field_info.nested_field
        idx = config.idx
        val = state_field_info.config_to_state(e.new)

        if state_field_info.column_idx is None:
            nve_data[nested_field].append(EntityTuple(idx, None, val))
        elif isinstance(state_field_info.column_idx, int):
            nve_data[nested_field].append(EntityTuple(idx, state_field_info.column_idx, val))
        else:
            for c, v in zip(state_field_info.column_idx, val):
                nve_data[nested_field].append(EntityTuple(idx, c, v))

    return nve_data


def nve_data_to_state_changes(nve_data, state):
    value_data = dict()
    for nf, nve_tuples in nve_data.items():
        nve_idx = sorted(list(set([int(t.idx) for t in nve_tuples])))
        row_map = {idx: i for i, idx in enumerate(nve_idx)}
        if nve_tuples[0].col is None:
            val = np.array(state.field(nf)[np.array(nve_idx)])
            col_map = None
            col_idx = None
        else:
            col_idx = sorted(list(set([t.col for t in nve_tuples])))
            col_map = {idx: i for i, idx in enumerate(col_idx)}
            val = np.array(state.field(nf)[np.ix_(state.row_idx(nf[0], nve_idx), col_idx)])
        value_data[nf] = ValueTuple(nve_idx, col_idx, row_map, col_map, val)

    state_changes = []
    for nf, value_tuple in value_data.items():
        for nve_tuple in nve_data[nf]:
            row = value_tuple.row_map[nve_tuple.idx]
            if nve_tuple.col is None:
                value_tuple.val[row] = nve_tuple.val
            else:
                col = value_tuple.col_map[nve_tuple.col]
                value_tuple.val[row, col] = nve_tuple.val
        state_changes.append(StateChangeTuple(nf, value_data[nf].nve_idx,
                                              value_data[nf].col_idx, value_tuple.val))

    return state_changes


def events_to_state_changes(events, state):

    nve_data = events_to_nve_data(events, state)
    return nve_data_to_state_changes(nve_data, state)


def rec_set_dataclass(var, nested_field, row_idx, column_idx, value):

    assert len(nested_field) > 0

    if isinstance(column_idx, int):
        column_idx = np.array([column_idx])

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
    n_entities_dict = {stype: len(config) for stype, config in config_dict.items() if stype != StateType.SIMULATOR}
    state = state or get_default_state(n_entities_dict)
    e_idx = jnp.zeros(sum(n_entities_dict.values()), dtype=int)
    for stype, configs in config_dict.items():
        params = configs[0].param_names()
        for p in params:
            state_field_info = configs_to_state_dict[stype][p]
            nve_idx = [c.idx for c in configs] if state_field_info.nested_field[0] == 'entity_state' else range(len(configs))
            change = rec_set_dataclass(state, state_field_info.nested_field, jnp.array(nve_idx), state_field_info.column_idx,
                                       jnp.array([state_field_info.config_to_state(getattr(c, p)) for c in configs]))
            state = state.set(**change)
        if stype.is_entity():
            e_idx.at[state.field(stype).nve_idx].set(jnp.array(range(n_entities_dict[stype])))

    # TODO: something weird with the to lines below, the second one will have no effect (would need state = state.set(.)), but if we fix it we get only zeros in entity_state.entitiy_idx. As it is it seems to get correct values though
    change = rec_set_dataclass(state, ('entity_state', 'entity_idx'), jnp.array(range(sum(n_entities_dict.values()))), None, e_idx)
    state.set(**change)

    return state


def set_configs_from_state(state, config_dict=None):
    if config_dict is None:
        config_dict = {stype: [] for stype in list(StateType)}
        for idx, stype_int in enumerate(state.entity_state.entity_type):
            stype = StateType(stype_int)
            config_dict[stype].append(stype_to_config[stype](idx=idx))
        config_dict[StateType.SIMULATOR].append(SimulatorConfig())
    for stype in config_dict.keys():
        for param, state_field_info in configs_to_state_dict[stype].items():
            value = state
            for f in state_field_info.nested_field:
                value = getattr(value, f)
            for config in config_dict[stype]:
                t = type(getattr(config, param))
                row_idx = state.row_idx(state_field_info.nested_field[0], config.idx)
                if state_field_info.column_idx is None:
                    value_to_set = value[row_idx]
                else:
                    value_to_set = value[row_idx, state_field_info.column_idx]
                value_to_set = state_field_info.state_to_config(value_to_set, t)
                config.param.update(**{param: value_to_set})
    return config_dict
