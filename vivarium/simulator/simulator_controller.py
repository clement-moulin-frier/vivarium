import param
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.simulator import config
import time, threading
from collections import namedtuple

param.Dynamic.time_dependent = True


class SimulatorController(param.Parameterized):

    client = param.Parameter(SimulatorGRPCClient())
    simulation_config = param.ClassSelector(config.SimulatorConfig, config.SimulatorConfig())
    agent_config = param.ClassSelector(config.AgentConfig, config.AgentConfig())
    selected_agents = param.ListSelector([0], objects=[0])
    refresh_change_period = param.Number(1)
    change_time = param.Integer(0)
    color = param.Color('#EEFF00')
    left_motor = param.Number(0., bounds=(0., 1.))
    right_motor = param.Number(0., bounds=(0., 1.))

    def __init__(self, **params):
        super().__init__(**params)
        self.pull_all_data()
        self.param.selected_agents.objects = range(self.simulation_config.n_agents)
        self.simulation_config.param.watch(self.push_simulation_config, self.simulation_config.export_fields, onlychanged=True) #, queued=True)
        self.agent_config.param.watch(self.push_agent_config, self.agent_config.export_fields, onlychanged=True) #, queued=True)
        self.param.watch(self.pull_agent_config, ['selected_agents'], onlychanged=True)
        self.param.watch(self.push_motors, ['left_motor', 'right_motor'], onlychanged=True)
        self.client.name = self.name
        threading.Thread(target=self._start_timer).start()

    def push_simulation_config(self, *events):
        print('push_simulation_config', self.simulation_config)
        d = {e.name: e.new for e in events}
        self.client.set_simulation_config(d)

    def push_agent_config(self, *events):
        print('push_agent_config', self.agent_config)
        d = {e.name: e.new for e in events}
        self.client.set_agent_config(self.selected_agents, d)

    def push_motors(self, *events):
        print(events)
        print('push_motors', {e.name for e in events})
        state = self.get_nve_state()
        motors = state.motor[self.selected_agents, :]
        for e in events:
            if e.name == 'left_motor':
                motor_idx = 0
            elif e.name == 'right_motor':
                motor_idx = 1
            else:
                raise(ValueError, 'events {e.name} not recognized')
            motors[:, motor_idx] = e.new
            self.client.set_state(self.selected_agents, ['motor'], motors)

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
        self.agent_config.param.update(**agent_config_dict)
        print(self.agent_config)

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
        return self.client.get_nve_state()

    def get_agent_configs(self):
        return self.client.get_agent_configs()

    # def set_motors(self, agent_idx, motors):
    #     return self.client.set_motors(agent_idx, motors)

if __name__ == "__main__":

    simulator = SimulatorController(client=SimulatorGRPCClient())
    MockEvent = namedtuple('MockEvent', ['name', 'new'])
    e = MockEvent(name='left_motor', new=0.1)
    simulator.push_motors(e)
