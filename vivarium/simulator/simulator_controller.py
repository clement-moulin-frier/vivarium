import param
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.simulator import config
from vivarium.simulator.sim_computation import EntityType
from vivarium import utils
import time
import threading
from contextlib import contextmanager
import numpy as np
import jax_md
from jax_md.rigid_body import RigidBody
from vivarium.simulator.sim_computation import AgentState
import math


param.Dynamic.time_dependent = True


class Selected(param.Parameterized):
    selection = param.ListSelector([0], objects=[0])

    def selection_nve_idx(self, nve_idx):
        return nve_idx[np.array(self.selection)].tolist()

class SimulatorController(param.Parameterized):

    client = param.Parameter(SimulatorGRPCClient())
    simulation_config = param.ClassSelector(config.SimulatorConfig, config.SimulatorConfig())
    entity_configs = param.Dict({EntityType.AGENT: config.AgentConfig(), EntityType.OBJECT: config.ObjectConfig()})
    selected_entities = param.Dict({EntityType.AGENT: Selected(), EntityType.OBJECT: Selected()})
    refresh_change_period = param.Number(1)
    change_time = param.Integer(0)

    def __init__(self, **params):
        super().__init__(**params)
        self.state = self.client.state
        self._entity_config_watchers = self.watch_entity_configs()
        self.update_entity_list()
        self.pull_all_data()
        self.simulation_config.param.watch(self.push_simulation_config, self.simulation_config.export_fields, onlychanged=True) #, queued=True)
        for etype, selected in self.selected_entities.items():
            selected.param.watch(self.pull_entity_configs, ['selection'], onlychanged=True, precedence=1)
        self.simulation_config.param.watch(self.update_entity_list, ['n_agents', 'n_objects'], onlychanged=True)
        self.client.name = self.name
        threading.Thread(target=self._start_timer).start()

    def watch_entity_configs(self):
        watchers = {etype: config.param.watch(self.push_state, config.export_fields, onlychanged=True)
                    for etype, config in self.entity_configs.items()}
        return watchers

    @contextmanager
    def dont_push_entity_configs(self):
        for etype, config in self.entity_configs.items():
            config.param.unwatch(self._entity_config_watchers[etype])
        try:
            yield None  #self.agent_config
        finally:
            self._entity_config_watchers = self.watch_entity_configs()

    def push_simulation_config(self, *events):
        print('push_simulation_config', self.simulation_config)
        d = {e.name: e.new for e in events}
        self.client.set_simulation_config(d)

    def push_state(self, *events):
        print('push_state')
        etype = config.config_to_etype[type(events[0].obj)]
        d = {e.name: e.new for e in events}
        selected_entities = self.selected_entities[etype]
        print('selected_entities in push_state', selected_entities.selection)
        for param, value in d.items():
            state_field_info = utils.configs_to_state_dict[etype][param]
            arr = np.tile(np.array(state_field_info.config_to_state(value)), (len(selected_entities.selection), 1))
            row_idx = selected_entities.selection if state_field_info.nested_field[0] != 'nve_state' else selected_entities.selection_nve_idx(getattr(self.state, f'{etype.name.lower()}_state').nve_idx)    # self.selected_nve_idx(self.state)[etype]
            self.client.set_state(state_field_info.nested_field,
                                  np.array(row_idx),
                                  state_field_info.column_idx,
                                  arr)

    def update_entity_list(self, *events):
        state = self.state
        for etype, selected in self.selected_entities.items():
            selected.param.selection.objects = state.entity_idx(etype).tolist()

    def pull_all_data(self):
        self.pull_entity_configs()
        self.pull_simulation_config()


    def pull_entity_configs(self, *events):
        state = self.state
        config_dict = {etype: [config] for etype, config in self.entity_configs.items()}

        with self.dont_push_entity_configs():
            for etype, selected in self.selected_entities.items():
                config_dict[etype][0].idx = selected.selection_nve_idx(getattr(self.state, f'{etype.name.lower()}_state').nve_idx)[0]  # selected[0]  # selected[0] if len(selected) > 0 else 0

            utils.set_configs_from_state(state, config_dict)
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

class Agent:
    def __init__(self, idx, state):
        self.idx = idx
        self.state = state
        self.subscribers = []
        self.behaviors = {}

    def sensors(self, key):
        return self.state.prox

    def subscribe(self, obj):
        self.subscribers.append(obj)

    def __getattr__(self, item):
        return self.state.item

    @property
    def motors(self):
        return self.state.motor

    @motors.setter
    def motors(self, value):
        for s in self.subscribers:
            s.notify(self.idx, 'motor', value)

    def attach_behavior(self, behavior_fn, name=None, weight=1.):
        self.behaviors[name or behavior_fn.__name__] = (behavior_fn, weight)

    def detach_behavior(self, name):
        del self.behaviors[name]

    def motors_from_behaviors(self):
        total_weights = 0.
        total_motor = np.zeros(2)
        for fn, w in self.behaviors.values():
            total_motor += w * np.array(fn(self))
            total_weights += w
        # print('motors_from_behaviors', total_motor, total_weights)
        print('motors', total_motor / total_weights)
        return total_motor / total_weights


class AgentList:
    def __init__(self, state):
        self.state = state
        n_agents = len(state.idx)
        self.agents = [Agent(ag, self.agent_state(ag)) for ag in range(n_agents)]

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
        super().__init__(**params)
        self.agent_list = AgentList(self.client.get_agent_state())
        self.agent_list.subscribe(self)
        self.from_stream = True

    @property
    def agents(self):
        state = self.client.state.agent_state if self.from_stream else self.client.get_agent_state()
        self.agent_list.update(state)
        return self.agent_list
        # return AgentList(self.client.get_nve_state())

    def notify(self, idx, field, value):
        # print('notify')
        self.client.set_state((field,), np.array([idx]), np.arange(2), value)

    def start_behavior(self, agent_idx, behavior_fn):
        self.client.start_behavior(agent_idx, behavior_fn)

    def run_all_behaviors(self, n_steps=math.inf):
        t = 0
        while t < n_steps:
            state = self.client.step()
            all_motor = np.zeros_like(state.agent_state.motor)
            for ag in self.agents.agents:
                all_motor[ag.idx, :] = ag.motors_from_behaviors()
            self.client.set_state(('agent_state', 'motor',), np.arange(all_motor.shape[0]), np.arange(all_motor.shape[1]), all_motor)
            t += 1



if __name__ == "__main__":

    # controller = SimulatorController()
    # controller.get_nve_state()
    # controller.selected_agents = [1]
    # print('idx = ', controller.agent_config.idx, 'y = ', controller.agent_config.y_position)

    controller = NotebookController()
    def behavior(prox):
        return [prox[1], prox[0]]

    for _ in range(30):
        controller.start_behavior(0, behavior)
