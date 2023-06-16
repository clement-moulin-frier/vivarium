import jax.numpy as jnp
from jax import jit
from jax import lax
import jax
import numpy as np

from jax_md import space, partition
from jax_md.util import f32

from contextlib import contextmanager

import vivarium.simulator.behaviors as behaviors
from vivarium.simulator.sim_computation import dynamics_rigid
from vivarium.simulator.config import AgentConfig, SimulatorConfig
from vivarium import utils

import time
import threading
import math
import param


class EngineConfig(param.Parameterized):
    simulation_config = param.ClassSelector(SimulatorConfig, instantiate=False)
    agent_configs = param.List(None)

    def __init__(self, **params):
        super().__init__(**params)
        self.agent_configs = self.agent_configs or [AgentConfig(idx=i, x_position=np.random.rand() * self.simulation_config.box_size,
                                                                y_position=np.random.rand() * self.simulation_config.box_size,
                                                                orientation=np.random.rand() * 2. * np.pi)
                                                    for i in range(self.simulation_config.n_agents)]
        # self.key = jax.random.PRNGKey(0)
        self.simulation_config.param.watch(self.update_space, ['box_size'], onlychanged=True, precedence=0)
        self.simulation_config.param.watch(self.init_state, ['box_size', 'n_agents'], onlychanged=True, precedence=1)
        self._neighbor_fn_watcher = self.simulation_config.param.watch(self.update_neighbor_fn, ['box_size', 'neighbor_radius'], onlychanged=True, precedence=2)
        self.simulation_config.param.watch(self.allocate_neighbors, ['n_agents'], onlychanged=True, precedence=2)
        self.simulation_config.param.watch(self.update_function_update, ['box_size', 'behavior_bank', 'dynamics_fn'], onlychanged=True, precedence=2)
        self._function_update_watcher = self.simulation_config.param.watch(self.update_function_update,
                                           ['dt', 'map_dim', 'to_jit'],
                                           onlychanged=True)
        for config in self.agent_configs:
            config.param.watch(self.update_state, list(config.to_dict().keys()), onlychanged=True)

        self.simulator = Simulator(self.simulation_config.box_size, self.simulation_config.map_dim,
                                   self.simulation_config.dt,  self.simulation_config.freq,
                                   self.simulation_config.use_fori_loop, self.simulation_config.num_steps_lax,
                                   self.simulation_config.neighbor_radius, self.simulation_config.to_jit,
                                   self.simulation_config.behavior_bank, self.simulation_config.dynamics_fn,
                                   **utils.get_init_state_kwargs(self.agent_configs))

    def update_space(self, event):
        print('update_space', event.name)
        assert event.name == 'box_size'
        for config in self.agent_configs:
            for pos in ['x_position', 'y_position']:
                if getattr(config, pos) >= self.simulation_config.box_size:
                    setattr(config, pos, getattr(config, pos) % self.simulation_config.box_size)

        self.simulator.update_space(box_size=event.new)


    def update_neighbor_fn(self, *events):
        print('_update_neighbor_fn', [e.name for e in events])
        kwargs = {name: getattr(self.simulation_config, name) for name in self._neighbor_fn_watcher.parameter_names}
        events_kwargs = {e.name: e.new for e in events}
        kwargs.update(events_kwargs)
        self.simulator.update_neighbor_fn(**kwargs)
        self.allocate_neighbors()

    # @param.depends('update_neighbor_fn', watch=True, on_init=False)
    def allocate_neighbors(self, *events):
        print('allocate_neighbors')
        # if self.simulator is not None:
        self.simulator.allocate_neighbors()

    def init_state(self, *events):
        print('init_state')
        self.simulator.init_state(**utils.get_init_state_kwargs(self.agent_configs))

    def update_state(self, *events):
        print('update_state')
        self.simulator.state = utils.set_state_from_agent_configs([e.obj for e in events], self.simulator.state,
                                                                      params=[e.name for e in events])

    def update_function_update(self, *events):
        print("_update_function_update", [e.name for e in events])
        kwargs = {name: getattr(self.simulation_config, name) for name in self._function_update_watcher.parameter_names}
        events_kwargs = {e.name: e.new for e in events}
        kwargs.update(events_kwargs)

        self.simulator.update_function_update(**kwargs)


class Simulator:
    def __init__(self, box_size, map_dim, dt, freq, use_fori_loop, num_steps_lax, neighbor_radius, to_jit,
                 behavior_bank, dynamics_fn,
                 **state_kwargs):

        self.behavior_bank = behavior_bank

        self.freq = freq
        self.use_fori_loop = use_fori_loop
        self.num_steps_lax = num_steps_lax
        self.dynamics_fn = dynamics_fn

        self.is_started = False
        self._to_stop = False
        self.key = jax.random.PRNGKey(0)

        self.update_space(box_size)
        self.update_function_update(map_dim, dt, to_jit)
        self.init_state(**state_kwargs)
        self.update_neighbor_fn(box_size, neighbor_radius)
        self.allocate_neighbors()


    def run(self, threaded=False):
        if self.is_started:
            raise Exception("Simulator is already started")
        if threaded:
            threading.Thread(target=self._run).start()
        else:
            return self._run()

    def _run(self, num_loops=math.inf):
        self.is_started = True
        print('Run starts')
        loop_count = 0
        while loop_count < num_loops:
            if self.freq is not None:
                time.sleep(1. / self.freq)
            if self._to_stop:
                self._to_stop = False
                break
            new_state = self.state
            neighbors = self.neighbors
            if self.use_fori_loop:
                new_state, neighbors = lax.fori_loop(0, self.num_steps_lax, self.update_fn,
                                                    (new_state, neighbors))
            else:

                for i in range(0, self.num_steps_lax):
                    new_state, neighbors = self.update_fn(i, (new_state, neighbors))
            # If the neighbor list can't fit in the allocation, rebuild it but bigger.
            if neighbors.did_buffer_overflow:
                print('REBUILDING')
                neighbors = self.neighbor_fn.allocate(new_state.position.center)
                # new_state, neighbors = lax.fori_loop(0, self.simulation_config.num_lax_loops, self.update_fn, (self.state, neighbors))
                for i in range(0, self.num_steps_lax):
                    new_state, neighbors = self.update_fn(i, (self.state, neighbors))
                assert not neighbors.did_buffer_overflow
            self.state = new_state
            self.neighbors = neighbors
            # print(self.state)
            loop_count += 1
        self.is_started = False
        print('Run stops')

    def set_motors(self, agent_idx, motor_idx, value):
        self.state = self.state.set(motor=self.state.motor.at[agent_idx, motor_idx].set(value))

    def set_state(self, agent_idx, nested_field, value):
        val_dim = value.shape
        column_idx = None if len(val_dim) == 1 else val_dim[1]
        change = utils.rec_set_dataclass(self.state, nested_field, column_idx, value)
        self.state = self.state.set(**change)

    def stop(self, blocking=True):
        self._to_stop = True
        if blocking:
            while self.is_started:
                time.sleep(0.01)
                print('still started')
            print('now stopped')

    @contextmanager
    def pause(self):
        self.stop(blocking=True)
        try:
            yield self
        finally:
            self.run(threaded=True)

    def update_space(self, box_size):
        self.displacement, self.shift = space.periodic(box_size)

    def update_function_update(self, map_dim, dt, to_jit, **kwargs):
        self.init_fn, self.step_fn = self.dynamics_fn(self.displacement, self.shift,
                                                      map_dim, dt,
                                                      self.behavior_bank)
        def update_fn(_, state_and_neighbors):
            state, neighs = state_and_neighbors
            neighs = neighs.update(state.position.center)
            return (self.step_fn(state=state, neighbor=neighs),
                    neighs)

        if to_jit:
            self.update_fn = jit(update_fn)
        else:
            self.update_fn = update_fn

    def init_state(self, **state_kwargs):
        self.state = self.init_fn(self.key, **state_kwargs)

    def update_neighbor_fn(self, box_size, neighbor_radius):
        self.neighbor_fn = partition.neighbor_list(self.displacement, box_size,
                                                   r_cutoff=neighbor_radius,
                                                   dr_threshold=10.,
                                                   capacity_multiplier=1.5,
                                                   format=partition.Sparse)

    def allocate_neighbors(self):
        self.neighbors = self.neighbor_fn.allocate(self.state.position.center)


if __name__ == "__main__":

    simulation_config = SimulatorConfig(to_jit=True, dynamics_fn=dynamics_rigid)

    engine_config = EngineConfig(simulation_config=simulation_config)

    engine_config.simulator.run()


