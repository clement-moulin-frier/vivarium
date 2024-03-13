import param

from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.controllers.config import SimulatorConfig
from vivarium.simulator.sim_computation import StateType
from vivarium.controllers import converters
import time
import threading
from contextlib import contextmanager
import logging

lg = logging.getLogger(__name__)

param.Dynamic.time_dependent = True


class SimulatorController(param.Parameterized):

    configs = param.Dict({StateType.SIMULATOR: SimulatorConfig(), StateType.AGENT: [], StateType.OBJECT: []})
    refresh_change_period = param.Number(1)
    change_time = param.Integer(0)

    def __init__(self, client=None, **params):
        super().__init__(**params)
        self.client = client or SimulatorGRPCClient()
        self.state = self.client.state
        configs_dict = converters.set_configs_from_state(self.state)
        for stype, configs in configs_dict.items():
            self.configs[stype] = configs
        self._config_watchers = self.watch_configs()
        self.pull_all_data()
        self.client.name = self.name
        self._in_batch = False

    def watch_configs(self):
        watchers = {etype: [config.param.watch(self.push_state, config.param_names(), onlychanged=True)
                            for config in configs]
                    for etype, configs in self.configs.items()}
        return watchers

    @property
    def simulator_config(self):
        return self.configs[StateType.SIMULATOR][0]

    def push_state(self, *events):
        if self._in_batch:
            self._event_list.extend(events)
            return
        lg.info('push_state %d', len(events))
        # lg.info(converters.events_to_state_changes(events))

        state_changes = converters.events_to_state_changes(events, self.state)
        for sc in state_changes:
            self.client.set_state(**sc._asdict())

    @contextmanager
    def dont_push_entity_configs(self):
        for etype, configs in self.configs.items():
            for i, config in enumerate(configs):
                config.param.unwatch(self._config_watchers[etype][i])
        try:
            yield None
        finally:
            self._config_watchers = self.watch_configs()

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

    def pull_all_data(self):
        self.pull_configs()

    def pull_configs(self, configs=None):
        configs = configs or self.configs
        state = self.state
        with self.dont_push_entity_configs():
            converters.set_configs_from_state(state, configs)
        return state

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


if __name__ == "__main__":

    controller = SimulatorController()
    controller.configs[StateType.AGENT][2].x_position = 1.
    lg.info(controller.client.get_state())

    lg.info('Done')
