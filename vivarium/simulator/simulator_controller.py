import param
from .simulator_client_abc import SimulatorClient
from .grpc_server.simulator_client import SimulatorGRPCClient
from . import config
import time, threading
import numpy as np

param.Dynamic.time_dependent = True

class SimulatorController(param.Parameterized):

    client = param.Parameter(SimulatorGRPCClient())
    simulation_config = param.ClassSelector(config.SimulatorConfig, config.SimulatorConfig())
    agent_config = param.ClassSelector(config.AgentConfig, config.AgentConfig())
    engine_config = param.ClassSelector(config.EngineConfig, config.EngineConfig())
    refresh_change_period = param.Number(1)
    change_time = param.Integer(0)
    # recorded_change_dict = param.Dict({})

    def __init__(self, **params):
        super().__init__(**params)
        # self.client = SimulatorGRPCClient()
        # self.simulation_config.param.watch_values(self._record_change, self.simulation_config.export_fields, queued=True)
        self.simulation_config.param.watch(self.push_config, self.simulation_config.export_fields, onlychanged=True, queued=True)
        self.client.name = self.name
        # self.recorded_change_dict = {}
        threading.Thread(target=self._start_timer).start()

    # @param.depends('simulation_config', watch=True)
    def push_config(self, *events):
        e = events[0]
        if len(events) == 1 and e.name == 'entity_behaviors' and np.array_equal(e.old, e.new):
            return
        print('push_config', self.simulation_config)
        # pcb_config.stop()
        #print(f"(event: {e.name} changed from {e.old} with type {type(e.old)} to {e.new} with type {type(e.new)}). Equals = {np.array_equal(e.old, e.new)}")
        self.client.set_simulation_config(self.simulation_config)
        # time.sleep(10)
        # pcb_config.start()




    def _start_timer(self):
        while True:
            change_time = self.client.get_change_time()
            if self.change_time < change_time:
                self.simulation_config.param.update(**self.client.get_recorded_changes())
                self.change_time = change_time
            # param.Dynamic.time_fn(self.change_time)
            # self.change_time = param.Dynamic.time_fn()
            time.sleep(self.refresh_change_period)

    # def _record_change(self, **kwargs):
    #     self.client.record_change(self.param.name, **kwargs)
    #     # self.recorded_change_dict.update(kwargs)

    # def get_recorded_changes(self):
    #     d = dict(self._recorded_change_dict)
    #     self.recorded_change_dict = {}
    #     return d

    @param.depends('simulation_config.displacement', 'simulation_config.box_size', 'agent_config.neighbor_radius',
                   watch=True, on_init=True)
    def _update_neighbor_fn(self):
        self.client.update_neighbor_fn()

    @param.depends('simulation_config.n_agents', 'simulation_config.box_size', watch=True, on_init=True)
    def _update_state_neighbors(self):
        self.client.update_state_neighbors()

    @param.depends('simulation_config.displacement', 'simulation_config.shift', 'simulation_config.map_dim',
                   'simulation_config.dt', 'agent_config.speed_mul', 'agent_config.theta_mul',
                   'agent_config.proxs_dist_max', 'agent_config.proxs_cos_min', 'agent_config.base_length',
                   'agent_config.wheel_diameter', 'simulation_config.entity_behaviors', 'engine_config.behavior_bank',
                   watch=True, on_init=True)
    def _update_function_update(self):
        self.client.update_function_update()

    @param.depends('simulation_config.n_agents', watch=True, on_init=True)
    def _update_behaviors(self):
        self.client.update_behaviors()


    def is_started(self):
        return self.client.is_started()
    def start(self):
        self.client.start()

    def stop(self):
        self.client.stop()

    def get_state(self):
        return self.client.get_state_arrays()