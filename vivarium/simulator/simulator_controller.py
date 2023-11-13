import param
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.simulator.config import config_to_etype, SimulatorConfig, AgentConfig, ObjectConfig
from vivarium.simulator.sim_computation import EntityType
from vivarium import utils
import time
import threading
from contextlib import contextmanager
import numpy as np
import jax_md
from jax_md.rigid_body import RigidBody
from vivarium.simulator.sim_computation import State, NVEState, AgentState, ObjectState
from vivarium.utils import set_configs_from_state
import math
from collections import defaultdict
from dataclasses import dataclass
from collections import namedtuple

param.Dynamic.time_dependent = True

NVETuple = namedtuple('NVETuple', ['idx', 'col', 'val'])
ValueTuple = namedtuple('ValueData', ['nve_idx', 'col_idx', 'row_map', 'col_map', 'val'])
StateChangeTuple = namedtuple('StateChange', ['nested_field', 'nve_idx', 'column_idx', 'value'])

def events_to_state_changes(events):
    nve_data = defaultdict(list)
    for e in events:
        config = e.obj
        param = e.name
        etype = config_to_etype[config.param.cls]
        idx = config.idx
        state_field_info = utils.configs_to_state_dict[etype][param]
        nested_field = state_field_info.nested_field
        val = state_field_info.config_to_state(e.new)
        # col = None if state_field_info.column_idx is None else state_field_info.column_idx[0]
        if state_field_info.column_idx is None:
            nve_data[nested_field].append(NVETuple(idx, None, val))
        else:
            if len(state_field_info.column_idx) == 1:
                val = [val]
            for c, v in zip(state_field_info.column_idx, val):
                nve_data[nested_field].append(NVETuple(idx, c, v))
        # if len(state_changes[nested_field]['col_idx']) == 0:
        #     if state_field_info.column_idx is None:
        #         state_changes[nested_field]['col_idx'] = None
        #     else:
        #         state_changes[nested_field]['col_idx'].append(state_field_info.column_idx[0])
    value_data = dict()
    for nf, nve_tuples in nve_data.items():
        nve_idx = sorted(list(set([t.idx for t in nve_tuples])))
        row_map = {idx: i for i, idx in enumerate(nve_idx)}
        if nve_tuples[0].col is None:
            val = np.zeros(len(nve_idx))
            col_map = None
            col_idx = None
        else:
            col_idx = sorted(list(set([t.col for t in nve_tuples])))
            col_map = {idx: i for i, idx in enumerate(col_idx)}
            val = np.zeros((len(nve_idx), len(col_idx)))
        value_data[nf] = ValueTuple(nve_idx, col_idx, row_map, col_map, val)

    state_changes = []
    for nf, value_tuple in value_data.items():
        # print('value_tuple.row_map', value_tuple.row_map)
        # print('value_tuple.col_map', value_tuple.col_map)
        for nve_tuple in nve_data[nf]:
            row = value_tuple.row_map[nve_tuple.idx]
            if nve_tuple.col is None:
                value_tuple.val[row] = nve_tuple.val
            else:
                col = value_tuple.col_map[nve_tuple.col]
                # print('nve_tuple.val', nve_tuple.val)
                value_tuple.val[row, col] = nve_tuple.val
        # print('nve_data[nf][0].col', nve_data[nf][0].col)
        state_changes.append(StateChangeTuple(nf, value_data[nf].nve_idx,
                                              value_data[nf].col_idx, value_tuple.val))

    return state_changes

    #     if args[1] is None:
    #         args[2] = np.zeros(len())
    #     l = args['nve_idx']
    #     sort_idx = sorted(range(len(l)), key=lambda k: l[k])
    #     args['row_value_idx'] = {idx: row for idx, row in zip(l, sort_idx)}
    #     args['nve_idx'] = [l[k] for k in sort_idx]
    #     n_row = len(args['nve_idx'])
    #     if args['col_idx'] is None:
    #         args['col_value_idx'] = None
    #         args['value'] = np.zeros(n_row)
    #     else:
    #         l = args['col_idx']
    #         sort_idx = sorted(range(len(l)), key=lambda k: l[k])
    #         args['col_value_idx'] = {idx: row for idx, row in zip(l, sort_idx)}
    #         args['col_idx'] = [l[k] for k in sort_idx]
    #         n_col = len(args['col_idx'])
    #         args['value'] = np.zeros((n_row, n_col))
    #
    # for e in events:
    #     state_field_info = utils.configs_to_state_dict[config_to_etype[config.param.cls]][e.name]
    #     nested_field = state_field_info.nested_field
    #     value = state_field_info.config_to_state(e.new)
    #     row = state_changes[nested_field]['row_value_idx'][e.obj.idx]
    #     if state_changes[nested_field]['col_value_idx'] is None:
    #         state_changes[nested_field]['value'][row] = value
    #     else:
    #         col = state_changes[nested_field]['col_value_idx'][e.obj.idx]
    #         state_changes[nested_field]['value'][row, col] = value
    #
    # return state_changes


class Selected(param.Parameterized):
    selection = param.ListSelector([0], objects=[0])

    def selection_nve_idx(self, nve_idx):
        return nve_idx[np.array(self.selection)].tolist()

class SimulatorController(param.Parameterized):

    client = param.Parameter(SimulatorGRPCClient())
    simulation_config = param.ClassSelector(SimulatorConfig, SimulatorConfig())
    entity_configs = param.Dict({EntityType.AGENT: [], EntityType.OBJECT: []})
    refresh_change_period = param.Number(1)
    change_time = param.Integer(0)

    def __init__(self, start_timer=True, **params):
        super().__init__(**params)
        self.state = self.client.state
        configs_dict = set_configs_from_state(self.state)
        for etype, configs in configs_dict.items():
            self.entity_configs[etype] = configs
        self._entity_config_watchers = self.watch_entity_configs()
        # self.update_entity_list()
        self.pull_all_data()
        self.simulation_config.param.watch(self.push_simulation_config, self.simulation_config.param_names(), onlychanged=True) #, queued=True)
        self.client.name = self.name
        self._in_batch = False
        if start_timer:
            threading.Thread(target=self._start_timer).start()

    def watch_entity_configs(self):
        watchers = {etype: [config.param.watch(self.push_state, config.param_names(), onlychanged=True) for config in configs]
                    for etype, configs in self.entity_configs.items()}
        return watchers

    def _push_state(self, configs, config_param):
        etype = config_to_etype[configs[0].param.cls]
        idx = np.array([c.idx for c in configs])
        state_field_info = utils.configs_to_state_dict[etype][config_param]
        arr = np.array([state_field_info.config_to_state(getattr(c, config_param)) for c in configs])
        # arr = np.array([state_field_info.config_to_state(v) for v in value])
        # row_idx = self.state.row_idx(state_field_info.nested_field[0], idx)
        self.client.set_state(state_field_info.nested_field,
                              idx,
                              state_field_info.column_idx,
                              arr)
    def push_state(self, *events):
        if self._in_batch:
            self._event_list.extend(events)
            return
        print('push_state', len(events))
        print(events_to_state_changes(events))

        state_changes = events_to_state_changes(events)
        for sc in state_changes:
            self.client.set_state(**sc._asdict())

        return

        per_param_change = defaultdict(list)
        for e in events:
            per_param_change[e.name].append(e.obj)
        for param, configs in per_param_change.items():
            self._push_state(configs, param)

        return

        for e in events:
            configs = [e.obj]
            param = e.name
            # value = e.new
            self._push_state(configs, param)
            # state_field_info = utils.configs_to_state_dict[etype][param]
            # arr = np.array(state_field_info.config_to_state(value))
            # row_idx = self.state.row_idx(state_field_info.nested_field[0], etype, entity_idx)
            # self.client.set_state(state_field_info.nested_field,
            #                       np.array([row_idx]),
            #                       state_field_info.column_idx,
            #                       arr)

    @contextmanager
    def dont_push_entity_configs(self):
        for etype, configs in self.entity_configs.items():
            for i, config in enumerate(configs):
                config.param.unwatch(self._entity_config_watchers[etype][i])
        try:
            yield None  #self.agent_config
        finally:
            self._entity_config_watchers = self.watch_entity_configs()

    @contextmanager
    def batch_set_state(self):
        self._in_batch = True
        self._event_list = []
        try:
            yield
        finally:
            print('exit batch_set_state', len(self._event_list))
            self._in_batch = False
            self.push_state(*self._event_list)
            self._event_list = None

    def push_simulation_config(self, *events):
        print('push_simulation_config', self.simulation_config)
        d = {e.name: e.new for e in events}
        self.client.set_simulation_config(d)



    def pull_all_data(self):
        self.pull_entity_configs()
        self.pull_simulation_config()


    def pull_entity_configs(self, *events):
        # print('pull_entity_configs', self.selected_entities[EntityType.AGENT].selection, self.entity_selected_configs[EntityType.AGENT].x_position)
        state = self.state
        # config_dict = {etype: [config] for etype, config in self.entity_configs.items()}

        with self.dont_push_entity_configs():
            # for etype, selected in self.selected_entities.items():
            #     config_dict[etype][0].idx = selected.selection_nve_idx(getattr(self.state, f'{etype.name.lower()}_state').nve_idx)[0]  # selected[0]  # selected[0] if len(selected) > 0 else 0

            utils.set_configs_from_state(state, self.entity_configs)

        return state


    def pull_simulation_config(self):
        sim_config_dict = self.client.get_sim_config().to_dict()
        self.simulation_config.param.update(**sim_config_dict)  # **self.client.get_recorded_changes())

    def pull_agent_config(self, *events):
        print('pull_agent_config')
        print(self.selected_agents)
        agent_config_dict = self.client.get_agent_config(self.selected_agents).to_dict()
        state = self.get_nve_state()
        with self.dont_push_agent_config():
            self.agent_config.param.update(**agent_config_dict)
            utils.set_agent_configs_from_state(state, [self.agent_config], ['position', 'prox', 'motor', 'behavior',
                                                                               'wheel_diameter', 'base_length',
                                                                               'speed_mul', 'theta_mul',
                                                                               'proxs_dist_max', 'proxs_cos_min',
                                                                               'entity_type'])
            # self.agent_config.update_from_state(self.state)
        print('updated_agent_config', agent_config_dict, self.agent_config.to_dict())

    def _start_timer(self):
        while True:
            change_time = self.client.get_change_time()
            if self.change_time < change_time:
                self.pull_all_data()
                self.change_time = change_time
            param.Dynamic.time_fn(self.change_time)
            self.change_time = param.Dynamic.time_fn()
            time.sleep(self.refresh_change_period)

    def is_started(self):
        return self.client.is_started()

    def start(self):
        self.client.start()

    def stop(self):
        self.client.stop()

    def update_state(self):
        self.state = self.client.get_state()
        return self.state

    def get_nve_state(self):
        self.state = self.client.get_nve_state()
        return self.state

    def get_agent_configs(self):
        configs = self.client.get_agent_configs()
        with param.parameterized.discard_events(self.agent_config):
            self.agent_config.param.update(**configs[self.selected_agents[0]].to_dict())
        return configs

    def add_agents(self, n_agents):
        _ = self.client.add_agents(n_agents, self.agent_config)

    def remove_agents(self):
        self.client.remove_agents(self.selected_agents)

class PanelController(SimulatorController):
    selected_entities = param.Dict({EntityType.AGENT: Selected(), EntityType.OBJECT: Selected()})
    selected_configs = param.Dict({EntityType.AGENT: AgentConfig(), EntityType.OBJECT: ObjectConfig()})
    def __init__(self, **params):
        self._selected_configs_watchers = None
        super().__init__(**params)
        self.update_entity_list()
        for etype, selected in self.selected_entities.items():
            selected.param.watch(self.pull_selected_configs, ['selection'], onlychanged=True, precedence=1)
        self.simulation_config.param.watch(self.update_entity_list, ['n_agents', 'n_objects'], onlychanged=True)
        #self.pull_all_data()
    def watch_selected_configs(self):
        watchers = {etype: config.param.watch(self.push_selected_to_state, config.param_names(), onlychanged=True)
                    for etype, config in self.selected_configs.items()}
        return watchers

    @contextmanager
    def dont_push_selected_configs(self):
        if self._selected_configs_watchers is not None:
            for etype, config in self.selected_configs.items():
                config.param.unwatch(self._selected_configs_watchers[etype])
        try:
            yield
        finally:
            self._selected_configs_watchers = self.watch_selected_configs()

    def update_entity_list(self, *events):
        state = self.state
        for etype, selected in self.selected_entities.items():
            selected.param.selection.objects = state.entity_idx(etype).tolist()

    def pull_selected_configs(self, *events):
        state = self.state
        config_dict = {etype: [config] for etype, config in self.selected_configs.items()}
        with self.dont_push_selected_configs():
            for etype, selected in self.selected_entities.items():
                config_dict[etype][0].idx = state.nve_idx(etype, selected.selection[0])  #_nve_idx(getattr(self.state, f'{etype.name.lower()}_state').nve_idx)[0]  # selected[0]  # selected[0] if len(selected) > 0 else 0
            utils.set_configs_from_state(state, config_dict)
        return state

    def pull_all_data(self):
        self.pull_selected_configs()
        self.pull_simulation_config()

    def push_selected_to_state(self, *events):
        print('push_state', len(events))
        # etype = config.config_to_etype[type(events[0].obj)]
        # d = {e.name: e.new for e in events}
        # print('selected_entities in push_state', selected_entities.selection)
        #for param, value in d.items():
        #with self.batch_set_state():
        for e in events:
            etype = config_to_etype[type(e.obj)]
            selected_entities = self.selected_entities[etype].selection
            for idx in selected_entities:
                setattr(self.entity_configs[etype][idx], e.name, e.new)
            # kwargs = self.entity_configs[etype].to_dict()
            # del kwargs['idx']
            # configs = [config.etype_to_config[etype](idx=idx, **kwargs) for idx in selected_entities]
            # param = e.name
            # print('selected_entities', selected_entities)
            # # value = np.tile(np.array(e.new), (len(selected_entities), 1))
            # self._push_state(configs, param)
            # # state_field_info = utils.configs_to_state_dict[etype][param]
            # # arr = np.tile(np.array(state_field_info.config_to_state(value)), (len(selected_entities.selection), 1))
            # # row_idx = selected_entities.selection if state_field_info.nested_field[0] != 'nve_state' else selected_entities.selection_nve_idx(getattr(self.state, f'{etype.name.lower()}_state').nve_idx)    # self.selected_nve_idx(self.state)[etype]
            # # self.client.set_state(state_field_info.nested_field,
            # #                       np.array(row_idx),
            # #                       state_field_info.column_idx,
            # #                       arr)
class StateField:
    def __init__(self, value, nested_fields):
        self.value = value
        self.nested_fields = nested_fields

class Entity:
    def __init__(self, config):
        self.config = config
        self.subscribers = []

        # nve_fields = [f.name for f in jax_md.dataclasses.fields(NVEState)]
        # for f in nve_fields:
        #     setattr(self, f, StateField(value=getattr(state.nve_state, f)[nve_idx], nested_fields=('nve_state', f)))
        # self.etype = EntityType(state.nve_state.entity_type[nve_idx])
        # fields = [f.name for f in jax_md.dataclasses.fields(type(state.field(self.etype)))]
        # for f in fields:
        #     setattr(self, f, StateField(value=getattr(state.field(self.etype), f)[state.nve_state.entity_idx[nve_idx]],
        #             nested_fields=(f'{self.etype.name.lower()}_state', f)))
        # # state_fields = [f.name for f in jax_md.dataclasses.fields(AgentState)]

    def __getattr__(self, item):
        if item in self.config.param_names():
            return getattr(self.config, item)
        else:
            return super().__getattr__(item)
    def __setattr__(self, item, val):
        if item != 'config' and item in self.config.param_names():
            return setattr(self.config, item, val)
        else:
            return super().__setattr__(item, val)
    def subscribe(self, obj):
        self.subscribers.append(obj)

class Agent(Entity):
    def __init__(self, config):
        super().__init__(config)
        self.config.behavior = 'manual'
        self.etype = EntityType.AGENT

        # self.idx = idx
        # self.state = state

        self.behaviors = {}

    def sensors(self):
        return [self.config.left_prox, self.config.right_prox]

    @property
    def motors(self):
        return self.state.agent_state.motor

    @motors.setter
    def motors(self, value):
        for s in self.subscribers:
            s.notify(self.idx, 'motor', value)

    def attach_behavior(self, behavior_fn, name=None, weight=1.):
        self.behaviors[name or behavior_fn.__name__] = (behavior_fn, weight)

    def detach_behavior(self, name):
        del self.behaviors[name]

    def detach_all_behaviors(self):
        self.behaviors = {}

    def behave(self):
        total_weights = 0.
        total_motor = np.zeros(2)
        for fn, w in self.behaviors.values():
            total_motor += w * np.array(fn(self))
            total_weights += w
        # print('motors_from_behaviors', total_motor, total_weights)
        motors = total_motor / total_weights
        self.left_motor, self.right_motor = motors
        # print('motors', total_motor / total_weights)
        # return total_motor / total_weights

    # def motors_from_behaviors(self):
    #     total_weights = 0.
    #     total_motor = np.zeros(2)
    #     for fn, w in self.behaviors.values():
    #         total_motor += w * np.array(fn(self))
    #         total_weights += w
    #     # print('motors_from_behaviors', total_motor, total_weights)
    #     print('motors', total_motor / total_weights)
    #     return total_motor / total_weights

class Object(Entity):
    def __init__(self, config):
        super().__init__(config)
        self.etype = EntityType.OBJECT

etype_to_class = {EntityType.AGENT: Agent, EntityType.OBJECT: Object}

class EntityList:
    def __init__(self, state):
        self.state = state
        # move code below as a method of sim_computation.State?

        self.entities = []
        #n_agents = len(state.idx)
        for etype in list(EntityType):
            eclass = etype_to_class[etype]
            field = etype.name.title()
            value = [eclass(i, state) for i in state.nve_idx(etype)]
            setattr(self, field, value)
            self.entities.extend(getattr(self, field))
        # self.agents = [Agent(ag, state) for ag in state.nve_idx(EntityType.AGENT)]
        # self.objects = [Object(ag, state) for ag in state.nve_idx(EntityType.OBJECT)]

    def __getitem__(self, item):
        return self.agents[item]

    def subscribe(self, obj):
        for ag in self.agents:
            ag.subscribe(obj)

    def update(self, state):
        self.state = state
        for ag in self.agents:
            ag.state = self.agent_state(ag.idx)

    def agent_state(self, item):
        state_fields = [f.name for f in jax_md.dataclasses.fields(AgentState)]
        state_kwargs = {}
        for field in state_fields:
            state_array = getattr(self.state, field)
            if isinstance(state_array, RigidBody):
                state_kwargs[field] = RigidBody(center=state_array[item], orientation=state_array[item])
            elif isinstance(state_array, np.ndarray):
                state_kwargs[field] = state_array[item]
            else:
                TypeError(f'state_kwargs has unknown field {type(state_kwargs)}')
        agent_state = AgentState(**state_kwargs)
        return agent_state  # Agent(item, agent_state)


class NotebookController(SimulatorController):

    def __init__(self, **params):
        super().__init__(start_timer=False, **params)
        # self.entity_list = EntityList(self.state)
        # # self.agent_list.subscribe(self)
        for etype in list(EntityType):
            setattr(self, f'{etype.name.lower()}s', [etype_to_class[etype](c) for c in self.entity_configs[etype]])
        self.from_stream = True
        self.simulation_config.freq = None

    # @property
    # def agents(self):
    #     state = self.client.state.agent_state if self.from_stream else self.client.get_agent_state()
    #     self.agent_list.update(state)
    #     return self.agent_list
    #     # return AgentList(self.client.get_nve_state())

    def notify(self, idx, field, value):
        # print('notify')
        self.client.set_state((field,), np.array([idx]), np.arange(2), value)

    def start_behavior(self, agent_idx, behavior_fn):
        self.client.start_behavior(agent_idx, behavior_fn)

    def run_all_behaviors(self, n_steps=math.inf):
        t = 0
        while t < n_steps:
            with self.batch_set_state():
                # all_motor = np.zeros_like(state.agent_state.motor)
                for ag in self.agents:
                    ag.behave()
                    # all_motor[ag.idx, :] = ag.motors_from_behaviors()
                # self.client.set_state(('agent_state', 'motor',), np.arange(all_motor.shape[0]), np.arange(all_motor.shape[1]), all_motor)
            self.state = self.client.step()
            print('state after step', self.state.agent_state.prox[0], self.state.agent_state.motor[0])
            # self.update_state()
            self.pull_entity_configs()

            print('config', self.agents[0].left_prox, self.agents[0].right_prox, self.agents[0].left_motor, self.agents[0].right_motor)
            print('state', self.state.agent_state.prox[0], self.state.agent_state.motor[0])
            # time.sleep(0.1)
            t += 1



if __name__ == "__main__":

    # controller = SimulatorController()
    # controller.get_nve_state()
    # controller.selected_agents = [1]
    # print('idx = ', controller.agent_config.idx, 'y = ', controller.agent_config.y_position)

    # controller = SimulatorController()
    # controller.entity_configs[EntityType.AGENT][2].x_position = 1.
    # print(controller.client.get_state())

    controller = SimulatorController()
    controller.selected_entities[EntityType.OBJECT].selection = [1]
    controller.selected_configs[EntityType.OBJECT].color = 'black'
    print(controller.update_state().object_state)

    #controller.selected_entities[EntityType.AGENT].selection = [1]
    print('Done')
    # def behavior(prox):
    #     return [prox[1], prox[0]]
    #
    # for _ in range(30):
    #     controller.start_behavior(0, behavior)
