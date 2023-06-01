import param
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.simulator import config
from vivarium import utils
import time, threading
from collections import namedtuple
from contextlib import contextmanager


param.Dynamic.time_dependent = True


class SimulatorController(param.Parameterized):

    client = param.Parameter(SimulatorGRPCClient())
    simulation_config = param.ClassSelector(config.SimulatorConfig, config.SimulatorConfig())
    agent_config = param.ClassSelector(config.AgentConfig, config.AgentConfig())
    selected_agents = param.ListSelector([0], objects=[0])
    refresh_change_period = param.Number(1)
    change_time = param.Integer(0)

    def __init__(self, **params):
        super().__init__(**params)
        self.state = None
        self._agent_config_watcher = self.watch_agent_config()
        self.pull_all_data()
        self.param.selected_agents.objects = range(self.simulation_config.n_agents)
        self.simulation_config.param.watch(self.push_simulation_config, self.simulation_config.export_fields, onlychanged=True) #, queued=True)
        self.param.watch(self.pull_agent_config, ['selected_agents'], onlychanged=True)
        self.client.name = self.name
        threading.Thread(target=self._start_timer).start()

    def watch_agent_config(self):
        return self.agent_config.param.watch(self.push_agent_config, self.agent_config.export_fields, onlychanged=True) #, queued=True)

    @contextmanager
    def dont_push_agent_config(self):
        self.agent_config.param.unwatch(self._agent_config_watcher)
        try:
            yield self.agent_config
        finally:
            self._agent_config_watcher = self.watch_agent_config()

    def push_simulation_config(self, *events):
        print('push_simulation_config', self.simulation_config)
        d = {e.name: e.new for e in events}
        self.client.set_simulation_config(d)

    def push_agent_config(self, *events):
        print('push_agent_config', self.agent_config)
        d = {e.name: e.new for e in events}
        self.client.set_agent_config(self.selected_agents, d)

    def pull_all_data(self):
        self.pull_agent_config()
        self.pull_simulation_config()

    def pull_simulation_config(self):
        sim_config_dict = self.client.get_sim_config().to_dict()
        self.simulation_config.param.update(**sim_config_dict)  # **self.client.get_recorded_changes())

    def pull_agent_config(self, *events):
        print('pull_agent_config')
        print(self.selected_agents)
        agent_config_dict = self.client.get_agent_config(self.selected_agents).to_dict()
        if self.state is None:
            self.get_nve_state()  ## Until the first time, then assumes that it is periodically called from elsewhere (e.g. from panel_app.py)
        with self.dont_push_agent_config():
            self.agent_config.param.update(**agent_config_dict)
            utils.set_agent_configs_from_state(self.state, [self.agent_config], ['position', 'prox', 'motor', 'behavior',
                                                                               'wheel_diameter', 'base_length',
                                                                               'speed_mul', 'theta_mul',
                                                                               'proxs_dist_max', 'proxs_cos_min',
                                                                               'entity_type'])
            # self.agent_config.update_from_state(self.state)
        print('updated_agent_config', self.agent_config.to_dict())

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

    def get_state(self):
        return self.client.get_state_arrays()

    def get_nve_state(self):
        self.state = self.client.get_nve_state()
        return self.state

    def get_agent_configs(self):
        configs = self.client.get_agent_configs()
        with param.parameterized.discard_events(self.agent_config):
            self.agent_config.param.update(**configs[self.selected_agents[0]].to_dict())
        return configs


if __name__ == "__main__":

    simulator = SimulatorController(client=SimulatorGRPCClient())
    MockEvent = namedtuple('MockEvent', ['name', 'new'])
    e = MockEvent(name='left_motor', new=0.1)
    simulator.push_motors(e)
