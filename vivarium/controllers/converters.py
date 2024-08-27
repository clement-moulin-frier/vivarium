import typing
import dataclasses
from collections import namedtuple, defaultdict

import jax_md
import jax.numpy as jnp
import numpy as np
import matplotlib.colors as mcolors

from vivarium.controllers.config import AgentConfig, ObjectConfig, SimulatorConfig, stype_to_config, config_to_stype
from vivarium.simulator.new_states import SimulatorState, AgentState, ObjectState, StateType
from vivarium.simulator.behaviors import behavior_name_map, reversed_behavior_name_map


# Define config fields for agents, objects and simulator
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
state_fields_dict = {
    StateType.AGENT: agent_state_fields,
    StateType.OBJECT: object_state_fields,
    StateType.SIMULATOR: simulator_state_fields
}


@dataclasses.dataclass
class StateFieldInfo:
    nested_field: typing.Tuple
    column_idx: np.array
    state_to_config: typing.Callable
    config_to_state: typing.Callable


# TODO : Add documentation here
identity_s_to_c = lambda x, typ: typ(x)
identity_c_to_s = lambda x: x
behavior_s_to_c = lambda x, typ: reversed_behavior_name_map[int(x)]
behavior_c_to_s = lambda x: behavior_name_map[x]
color_s_to_c = lambda x, typ: mcolors.to_hex(np.array(x))  # Warning : temporary (below as well)
color_c_to_s = lambda x: mcolors.to_rgb(x)
mass_center_s_to_c = lambda x, typ: typ(x)
mass_center_c_to_s = lambda x: [x]
exists_c_to_s = lambda x: int(x)
neighbor_map_s_to_c = lambda x, typ: x


# Define conversions between agents configs and state dictionary
agent_configs_to_state_dict = {
    'x_position': StateFieldInfo(('entity_state', 'position', 'center'), 0, identity_s_to_c, identity_c_to_s),
    'y_position': StateFieldInfo(('entity_state', 'position', 'center'), 1, identity_s_to_c, identity_c_to_s),
    'orientation': StateFieldInfo(('entity_state', 'position', 'orientation'), None, identity_s_to_c, identity_c_to_s),
    'mass_center': StateFieldInfo(('entity_state', 'mass', 'center'), np.array([0]), mass_center_s_to_c, mass_center_c_to_s),
    'mass_orientation': StateFieldInfo(('entity_state', 'mass', 'orientation'), None, identity_s_to_c, identity_c_to_s),
    'diameter': StateFieldInfo(('entity_state', 'diameter'), None, identity_s_to_c, identity_c_to_s),
    'friction': StateFieldInfo(('entity_state', 'friction'), None, identity_s_to_c, identity_c_to_s),
    'left_motor': StateFieldInfo(('agent_state', 'motor',), 0, identity_s_to_c, identity_c_to_s),
    # TODO : Might need to change this logic where you can have a list of proxs for different behaviors ...
    'right_motor': StateFieldInfo(('agent_state', 'motor',), 1, identity_s_to_c, identity_c_to_s),
    'left_prox': StateFieldInfo(('agent_state', 'prox',), 0, identity_s_to_c, identity_c_to_s),
    'right_prox': StateFieldInfo(('agent_state', 'prox',), 1, identity_s_to_c, identity_c_to_s),
    'proximity_map_dist': StateFieldInfo(('agent_state', 'proximity_map_dist',), slice(None), neighbor_map_s_to_c, identity_c_to_s),
    'proximity_map_theta': StateFieldInfo(('agent_state', 'proximity_map_theta',), slice(None), neighbor_map_s_to_c, identity_c_to_s),
    # TODO : Think params and sensed should be like that but not sure (because it returns a list and not just a single value)
    'params': StateFieldInfo(('agent_state', 'params',), slice(None), neighbor_map_s_to_c, identity_c_to_s),
    'sensed': StateFieldInfo(('agent_state', 'sensed',), slice(None), neighbor_map_s_to_c, identity_c_to_s),
    'behavior': StateFieldInfo(('agent_state', 'behavior',), slice(None), neighbor_map_s_to_c, identity_c_to_s),
    #'behavior': StateFieldInfo(('agent_state', 'behavior',), None, behavior_s_to_c, behavior_c_to_s),
    'color': StateFieldInfo(('agent_state', 'color',), np.arange(3), color_s_to_c, color_c_to_s),
    'idx': StateFieldInfo(('agent_state', 'ent_idx',), None, identity_s_to_c, identity_c_to_s),
    'exists': StateFieldInfo(('entity_state', 'exists'), None, identity_s_to_c, exists_c_to_s)
}

agent_configs_to_state_dict.update(
    {f: StateFieldInfo(('agent_state', f,), None, identity_s_to_c, identity_c_to_s) for f in agent_common_fields if f not in agent_configs_to_state_dict}
)

# Define conversions between objects configs and state dictionary
object_configs_to_state_dict = {
    'x_position': StateFieldInfo(('entity_state', 'position', 'center'), 0, identity_s_to_c, identity_c_to_s),
    'y_position': StateFieldInfo(('entity_state', 'position', 'center'), 1, identity_s_to_c, identity_c_to_s),
    'orientation': StateFieldInfo(('entity_state', 'position', 'orientation'), None, identity_s_to_c, identity_c_to_s),
    'mass_center': StateFieldInfo(('entity_state', 'mass', 'center'), np.array([0]), mass_center_s_to_c, mass_center_c_to_s),
    'mass_orientation': StateFieldInfo(('entity_state', 'mass', 'orientation'), None, identity_s_to_c, identity_c_to_s),
    'diameter': StateFieldInfo(('entity_state', 'diameter'), None, identity_s_to_c, identity_c_to_s),
    'friction': StateFieldInfo(('entity_state', 'friction'), None, identity_s_to_c, identity_c_to_s),
    'color': StateFieldInfo(('object_state', 'color',), np.arange(3), color_s_to_c, color_c_to_s),
    'idx': StateFieldInfo(('object_state', 'ent_idx',), None, identity_s_to_c, identity_c_to_s),
    'exists': StateFieldInfo(('entity_state', 'exists'), None, identity_s_to_c, exists_c_to_s)
}

object_configs_to_state_dict.update(
    {f: StateFieldInfo(('object_state', f,), None, identity_s_to_c, identity_c_to_s) for f in object_common_fields if f not in object_configs_to_state_dict}
)

simulator_configs_to_state_dict = {}
simulator_configs_to_state_dict.update({f: StateFieldInfo(('simulator_state', f,), None, identity_s_to_c, identity_c_to_s) for f in simulator_common_fields if f not in simulator_configs_to_state_dict})

# Define conversions between objects configs and state dictionary
configs_to_state_dict = {
    StateType.AGENT: agent_configs_to_state_dict,
    StateType.OBJECT: object_configs_to_state_dict,
    StateType.SIMULATOR: simulator_configs_to_state_dict
}


EntityTuple = namedtuple('EntityTuple', ['idx', 'col', 'val'])
ValueTuple = namedtuple('ValueData', ['ent_idx', 'col_idx', 'row_map', 'col_map', 'val'])
StateChangeTuple = namedtuple('StateChange', ['nested_field', 'ent_idx', 'column_idx', 'value'])


# TODO : Add documentation
def events_to_nve_data(events, state):
    nve_data = defaultdict(list)
    print(f"{events = }")
    print("")
    for e in events:
        config = e.obj
        param = e.name
        etype = config_to_stype[config.param.cls]
        state_field_info = configs_to_state_dict[etype][param]
        nested_field = state_field_info.nested_field
        idx = config.idx
        val = state_field_info.config_to_state(e.new)

        print("\n Slice state_field_info.column_idx")
        print(f"{state_field_info.column_idx = }")
        print(f"{val = }")
        if state_field_info.column_idx is None:
            print("NONE")
            nve_data[nested_field].append(EntityTuple(idx, None, val))
        elif isinstance(state_field_info.column_idx, int):
            print("INT")
            nve_data[nested_field].append(EntityTuple(idx, state_field_info.column_idx, val))
        elif isinstance(state_field_info.column_idx, slice):
            print("SLICE")
            print(f"{state_field_info.column_idx = }")
            print(f"{val = }")
            nve_data[nested_field].append(EntityTuple(idx, state_field_info.column_idx, val))
        else:
            for c, v in zip(state_field_info.column_idx, val):
                nve_data[nested_field].append(EntityTuple(idx, c, v))
    
    print("\nfinal nve data")
    print(f"{nve_data = }")
    return nve_data


# TODO : Add documentation
def nve_data_to_state_changes(nve_data, state):
    value_data = dict()
    for nf, nve_tuples in nve_data.items():
        for t in nve_tuples:
            print(f"{t = }")
            print(f"{t.idx = }")
        ent_idx = sorted(list(set([int(t.idx) for t in nve_tuples])))
        row_map = {idx: i for i, idx in enumerate(ent_idx)}
        print(f"{t.col = }")
        if nve_tuples[0].col is None:
            val = np.array(state.field(nf)[np.array(ent_idx)])
            col_map = None
            col_idx = None
        # TODO : Added a condition for slice here
        elif isinstance(nve_tuples[0].col, slice):
            val = np.array(state.field(nf)[np.array(ent_idx)])
            col_map = nve_tuples[0].col
            # col_map = None
            col_idx = None
        else:
            col_idx = sorted(list(set([t.col for t in nve_tuples])))
            col_map = {idx: i for i, idx in enumerate(col_idx)}
            val = np.array(state.field(nf)[np.ix_(state.row_idx(nf[0], ent_idx), col_idx)])
        value_data[nf] = ValueTuple(ent_idx, col_idx, row_map, col_map, val)

    print(f"\n{value_data = }")
    state_changes = []
    for nf, value_tuple in value_data.items():
        for nve_tuple in nve_data[nf]:
            row = value_tuple.row_map[nve_tuple.idx]
            if nve_tuple.col is None:
                value_tuple.val[row] = nve_tuple.val
            # TODO : Add a condition for slice here
            elif isinstance(nve_tuple.col, slice):
                print("SLICE")
                value_tuple.val[row] = nve_tuple.val
            else:
                col = value_tuple.col_map[nve_tuple.col]
                value_tuple.val[row, col] = nve_tuple.val
        state_changes.append(StateChangeTuple(
            nf, value_data[nf].ent_idx,
            value_data[nf].col_idx, value_tuple.val
        ))

    print(f"\n{state_changes = }")
    return state_changes


def events_to_state_changes(events, state):
    nve_data = events_to_nve_data(events, state)
    return nve_data_to_state_changes(nve_data, state)


# TODO : Add documentation
def rec_set_dataclass(var, nested_field, row_idx, column_idx, value):
    assert len(nested_field) > 0
    if isinstance(column_idx, int):
        column_idx = np.array([column_idx])

    if len(nested_field) == 1:
        field = nested_field[0]
        if isinstance(column_idx, slice):
            column_idx = np.arange(
                column_idx.start if column_idx.start is not None else 0,
                column_idx.stop if column_idx.stop is not None else getattr(var, field).shape[1],
                column_idx.step if column_idx.step is not None else 1, dtype=int
            )
        if column_idx is None or len(column_idx) == 0:
            d = {field: getattr(var, field).at[row_idx].set(value.reshape(-1))}
        else:
            d = {field: getattr(var, field).at[jnp.ix_(row_idx, column_idx)].set(value.reshape(len(row_idx), len(column_idx)))}
        return d
    else:
        next_var = getattr(var, nested_field[0])
        d = rec_set_dataclass(next_var, nested_field[1:], row_idx, column_idx, value)
        return {nested_field[0]: next_var.set(**d)}


# TODO : Add documentation
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
