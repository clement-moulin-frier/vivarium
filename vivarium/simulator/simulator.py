import jax.numpy as jnp
from jax import jit, grad
from jax import lax
import jax
import numpy as np
from jax_md import space, partition

from collections import namedtuple
from contextlib import contextmanager


import vivarium.simulator.behaviors as behaviors
from vivarium.simulator.sim_computation import dynamics_rigid
from vivarium.simulator.config import AgentConfig, SimulatorConfig

import time
import threading
import math
import param


def generate_positions_orientations(key, n_agents, box_size):
    key, subkey = jax.random.split(key)
    positions = box_size * jax.random.uniform(subkey, (n_agents, 2))
    key, subkey = jax.random.split(key)
    orientations = jax.random.uniform(subkey, (n_agents,), maxval=2 * np.pi)
    return key, positions, orientations


class EngineConfig(param.Parameterized):
    simulation_config = param.ClassSelector(SimulatorConfig, instantiate=False)
    agent_configs = param.List(None)
    dynamics_fn = param.Parameter()
    displacement = param.Parameter()
    shift = param.Parameter()
    behavior_bank = param.List(behaviors.behavior_bank)
    behavior_name_map = param.Dict(behaviors.behavior_name_map)
    neighbor_fn = param.Parameter()
    state = param.Parameter(None)


    def __init__(self, **params):
        super().__init__(**params)
        self.agent_configs = self.agent_configs or [AgentConfig() for _ in range(self.simulation_config.n_agents)]
        self.key = key = jax.random.PRNGKey(0)
        self.simulation_config.param.watch(self.update_space, ['box_size'], onlychanged=True, precedence=0)
        self.param.watch(self.update_neighbor_fn, ['displacement'], onlychanged=True, precedence=1)
        self.simulation_config.param.watch(self.update_state, ['box_size'], onlychanged=True)
        self.simulation_config.param.watch(self.update_neighbor_fn, ['neighbor_radius'], onlychanged=True)
        self.param.watch(self.update_neighbors, ['neighbor_fn'], onlychanged=True, precedence=3)
        self.param.watch(self.update_function_update, ['displacement', 'shift', 'behavior_bank'], onlychanged=True, precedence=2)
        self.simulation_config.param.watch(self.update_function_update,
                                           ['dt', 'map_dim', 'to_jit'],
                                           onlychanged=True)
        for config in self.agent_configs:
            config.param.watch(self.update_state, list(config.to_dict().keys()), onlychanged=True)

        MockEvent = namedtuple('MockEvent', ['name', 'new'])
        with param.parameterized.discard_events(self):
            self.update_space(MockEvent(name='box_size', new=self.simulation_config.box_size))
            self.update_function_update()
            self.update_state()
            self.update_neighbor_fn()
            self.update_neighbors()

    def update_space(self, event):
        print('update_space', event.name)
        self.displacement, self.shift = space.periodic(event.new)


    def update_neighbor_fn(self, *events):
        print('_update_neighbor_fn', [e.name for e in events])
        self.neighbor_fn = partition.neighbor_list(self.displacement,
                                                   self.simulation_config.box_size,
                                                   r_cutoff=self.simulation_config.neighbor_radius,
                                                   dr_threshold=10.,
                                                   capacity_multiplier=1.5,
                                                   format=partition.Sparse)

    def update_state(self, *events):
        print('update_state',  [e.name for e in events])

        if len(events) == 0:  # kind of hack: when we initialize the state for the first time
            self.key, subkey = jax.random.split(self.key)
            self.key, positions, orientations = generate_positions_orientations(key=self.key,
                                                                      n_agents=len(self.agent_configs),
                                                                      box_size=self.simulation_config.box_size)

            self.state = self.init_fn(self.key, positions=positions, orientations=orientations,
                                      agent_configs_as_array_dict=self.agent_configs_as_array_dict())
        else:  # when a change is made from the interface or controller
            self.state = self.state.set(**self.agent_configs_as_array_dict())

    def update_neighbors(self, *events):
        if self.state is None:
            return
        print('update_neighbors',  [e.name for e in events])
        for e in events:
            if e.name == 'n_agents':
                assert self.state.position.center.shape[0] == e.new

        self.neighbors = self.neighbor_fn.allocate(self.state.position.center)

    def update_function_update(self, *events):
        print("_update_function_update", [e.name for e in events])
        self.init_fn, self.step_fn = self.dynamics_fn(self.displacement, self.shift,
                                                      self.simulation_config.map_dim, self.simulation_config.dt,
                                                      self.behavior_bank)

        def update_fn(_, state_and_neighbors):
            state, neighs = state_and_neighbors
            neighs = neighs.update(state.position.center)
            return (self.step_fn(state=state, neighbor=neighs),
                    neighs)

        if self.simulation_config.to_jit:
            self.update_fn = jit(update_fn)
        else:
            self.update_fn = update_fn

    def update_behaviors(self, *events):
        print('update_behaviors', [e.name for e in events])
        self.entity_behaviors = jnp.array([self.behavior_name_map[config.behavior.name] for config in self.simulation_config.agent_configs], dtype=int)

    def agent_configs_as_array_dict(self):
        keys = self.agent_configs[0].to_dict().keys()
        d = {}
        for k in keys:
            if k == 'behavior':
                d[k] = jnp.array([self.behavior_name_map[config.behavior]
                                  for config in self.agent_configs], dtype=int)
            else:
                dtype = type(getattr(self.agent_configs[0], k))
                d[k] = jnp.array([getattr(config, k) for config in self.agent_configs], dtype=dtype)
        return d

    def from_array_dict(self, array_dict):
        for i, config in enumerate(self.agent_configs):
            ag_dict = {k: v[i] for k, v in array_dict}
            config.param.update(**ag_dict)


class Simulator():

    def __init__(self, engine_config):
        self.engine_config = engine_config
        self.simulation_config = self.engine_config.simulation_config
        self.agent_configs = self.engine_config.agent_configs
        self._state = self.engine_config.state
        self.neighbors = self.engine_config.neighbors
        self.is_started = False
        self._to_stop = False

    @property
    def state(self):
        return self.engine_config.state

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
            if self.simulation_config.freq is not None:
                time.sleep(1. / self.simulation_config.freq)
            if self._to_stop:
                self._to_stop = False
                break
            if self.simulation_config.use_fori_loop:
                new_state, neighbors = lax.fori_loop(0, self.simulation_config.num_steps_lax, self.update_fn,
                                                    (self._state, self.neighbors))
            else:
                new_state = self.engine_config.state
                neighbors = self.engine_config.neighbors
                for i in range(0, self.simulation_config.num_steps_lax):
                    new_state, neighbors = self.engine_config.update_fn(i, (new_state, neighbors))
            # If the neighbor list can't fit in the allocation, rebuild it but bigger.
            if neighbors.did_buffer_overflow:
                print('REBUILDING')
                neighbors = self.simulation_config.neighbor_fn.allocate(self._state.positions)
                new_state, neighbors = lax.fori_loop(0, self.simulation_config.num_lax_loops, self.update_fn, (self._state, neighbors))
                assert not neighbors.did_buffer_overflow
            self.engine_config.state = new_state
            self.engine_config.neighbors = neighbors
            loop_count += 1
        self.is_started = False
        print('Run stops')

    def set_motors(self, agent_idx, motor_idx, value):
        self.engine_config.state = self.state.set(motor=self.engine_config.state.motor.at[agent_idx, motor_idx].set(value))

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


if __name__ == "__main__":

    simulation_config = SimulatorConfig(to_jit=True)

    engine_config = EngineConfig(simulation_config=simulation_config, dynamics_fn=dynamics_rigid)

    simulator = Simulator(engine_config=engine_config)

    simulator.run()

