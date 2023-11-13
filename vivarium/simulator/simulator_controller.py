import param
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.simulator.config import config_to_etype, SimulatorConfig, AgentConfig, ObjectConfig
from vivarium.simulator.sim_computation import EntityType
from vivarium import utils
import time
import threading
from contextlib import contextmanager
import numpy as np
from vivarium.utils import set_configs_from_state
import math

param.Dynamic.time_dependent = True


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
        self.pull_all_data()
        self.simulation_config.param.watch(self.push_simulation_config, self.simulation_config.param_names(),
                                           onlychanged=True)
        self.client.name = self.name
        self._in_batch = False
        if start_timer:
            threading.Thread(target=self._start_timer).start()

    def watch_entity_configs(self):
        watchers = {etype: [config.param.watch(self.push_state, config.param_names(), onlychanged=True)
                            for config in configs]
                    for etype, configs in self.entity_configs.items()}
        return watchers

    def push_state(self, *events):
        if self._in_batch:
            self._event_list.extend(events)
            return
        print('push_state', len(events))
        print(utils.events_to_state_changes(events))

        state_changes = utils.events_to_state_changes(events)
        for sc in state_changes:
            self.client.set_state(**sc._asdict())

    @contextmanager
    def dont_push_entity_configs(self):
        for etype, configs in self.entity_configs.items():
            for i, config in enumerate(configs):
                config.param.unwatch(self._entity_config_watchers[etype][i])
        try:
            yield None
        finally:
            self._entity_config_watchers = self.watch_entity_configs()

    @contextmanager
    def batch_set_state(self):
        self._in_batch = True
        self._event_list = []
        try:
            yield
        finally:
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
        state = self.state
        with self.dont_push_entity_configs():
            utils.set_configs_from_state(state, self.entity_configs)
        return state

    def pull_simulation_config(self):
        sim_config_dict = self.client.get_sim_config().to_dict()
        self.simulation_config.param.update(**sim_config_dict)

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
                config_dict[etype][0].idx = state.nve_idx(etype, selected.selection[0])
            utils.set_configs_from_state(state, config_dict)
        return state

    def pull_all_data(self):
        self.pull_selected_configs()
        self.pull_simulation_config()

    def push_selected_to_state(self, *events):
        print('push_selected_to_state', len(events))
        for e in events:
            etype = config_to_etype[type(e.obj)]
            selected_entities = self.selected_entities[etype].selection
            for idx in selected_entities:
                setattr(self.entity_configs[etype][idx], e.name, e.new)


class Entity:
    def __init__(self, config):
        self.config = config
        self.subscribers = []

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

        self.behaviors = {}

    def sensors(self):
        return [self.config.left_prox, self.config.right_prox]

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
        motors = total_motor / total_weights
        self.left_motor, self.right_motor = motors


class Object(Entity):
    def __init__(self, config):
        super().__init__(config)
        self.etype = EntityType.OBJECT


etype_to_class = {EntityType.AGENT: Agent, EntityType.OBJECT: Object}


class NotebookController(SimulatorController):

    def __init__(self, **params):
        super().__init__(start_timer=False, **params)
        for etype in list(EntityType):
            setattr(self, f'{etype.name.lower()}s', [etype_to_class[etype](c) for c in self.entity_configs[etype]])
        self.from_stream = True
        self.simulation_config.freq = None

    def start_behavior(self, agent_idx, behavior_fn):
        self.client.start_behavior(agent_idx, behavior_fn)

    def run_all_behaviors(self, n_steps=math.inf):
        t = 0
        while t < n_steps:
            with self.batch_set_state():
                for ag in self.agents:
                    ag.behave()
            self.state = self.client.step()
            self.pull_entity_configs()

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
