import param

from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.controllers.config import config_to_etype, SimulatorConfig, AgentConfig, ObjectConfig
from vivarium.simulator.sim_computation import EntityType
from vivarium.controllers import converters
import time
import threading
from contextlib import contextmanager

param.Dynamic.time_dependent = True


class SimulatorController(param.Parameterized):

    client = param.Parameter(SimulatorGRPCClient())
    simulation_config = param.ClassSelector(SimulatorConfig, SimulatorConfig())
    entity_configs = param.Dict({EntityType.AGENT: [], EntityType.OBJECT: []})
    refresh_change_period = param.Number(1)
    change_time = param.Integer(0)

    def __init__(self, start_timer=True, **params):
        super().__init__(**params)
        self.state = self.client.state
        configs_dict = converters.set_configs_from_state(self.state)
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
        # print(converters.events_to_state_changes(events))

        state_changes = converters.events_to_state_changes(events, self.state)
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
            converters.set_configs_from_state(state, self.entity_configs)
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


if __name__ == "__main__":

    controller = SimulatorController()
    controller.entity_configs[EntityType.AGENT][2].x_position = 1.
    print(controller.client.get_state())

    print('Done')
