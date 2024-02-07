from jax import jit
from jax import lax
import jax
import jax.numpy as jnp
import numpy as np

from jax_md import space, partition, dataclasses

from contextlib import contextmanager

from vivarium.simulator.sim_computation import dynamics_rigid, EntityType, StateType, SimulatorState
from vivarium.controllers.config import AgentConfig, ObjectConfig, SimulatorConfig
from vivarium.controllers import converters
import vivarium.simulator.behaviors as behaviors

import time
import threading
import math


class Simulator:
    def __init__(self, state, behavior_bank, dynamics_fn):

        self.state = state
        self.behavior_bank = behavior_bank
        self.dynamics_fn = dynamics_fn

        all_attrs = [f.name for f in dataclasses.fields(SimulatorState)]
        for attr in all_attrs:
            self.update_attr(attr, SimulatorState.get_type(attr))

        self._is_started = False
        self._to_stop = False
        self.key = jax.random.PRNGKey(0)

        self.update_space(self.box_size)
        self.update_function_update()
        self.init_state(state)
        self.update_neighbor_fn(self.box_size, self.neighbor_radius)
        self.allocate_neighbors()

    def run(self, threaded=False, num_loops=math.inf):
        if self._is_started:
            raise Exception("Simulator is already started")
        if threaded:
            threading.Thread(target=self._run).start()
        else:
            return self._run(num_loops)

    def _run(self, num_loops=math.inf):
        self._is_started = True
        print('Run starts')
        loop_count = 0
        while loop_count < num_loops:
            if self._to_stop:
                self._to_stop = False
                break
            if float(self.freq) > 0.:
                time.sleep(1. / float(self.freq))
            new_state = self.state
            neighbors = self.neighbors
            if self.state.simulator_state.use_fori_loop:
                new_state, neighbors = lax.fori_loop(0, self.num_steps_lax, self.update_fn,
                                                    (new_state, neighbors))
            else:

                for i in range(0, self.num_steps_lax):
                    new_state, neighbors = self.update_fn(i, (new_state, neighbors))
            # If the neighbor list can't fit in the allocation, rebuild it but bigger.
            if neighbors.did_buffer_overflow:
                print('REBUILDING')
                neighbors = self.allocate_neighbors(new_state.nve_state.position.center)
                # new_state, neighbors = lax.fori_loop(0, self.simulation_config.num_lax_loops, self.update_fn, (self.state, neighbors))
                for i in range(0, self.num_steps_lax):
                    new_state, neighbors = self.update_fn(i, (self.state, neighbors))
                assert not neighbors.did_buffer_overflow
            self.state = new_state
            self.neighbors = neighbors
            # print(self.state)
            loop_count += 1
        self._is_started = False
        print('Run stops')

    def set_state(self, nested_field, nve_idx, column_idx, value):
        print('set_state', nested_field, nve_idx, column_idx, value)
        row_idx = self.state.row_idx(nested_field[0], jnp.array(nve_idx))
        col_idx = None if column_idx is None else jnp.array(column_idx)
        change = converters.rec_set_dataclass(self.state, nested_field, row_idx, col_idx, value)
        self.state = self.state.set(**change)

        if nested_field[0] == 'simulator_state':
            self.update_attr(nested_field[1], SimulatorState.get_type(nested_field[1]))

        if nested_field == ('simulator_state', 'box_size'):
            self.update_space(self.box_size)

        if nested_field == ('simulator_state', 'box_size') or nested_field == ('simulator_state', 'neighbor_radius'):
            self.update_neighbor_fn(box_size=self.box_size,
                                    neighbor_radius=self.neighbor_radius)

        if nested_field == ('simulator_state', 'box_size') or nested_field == ('simulator_state', 'dt') or \
                nested_field == ('simulator_state', 'to_jit'):
            self.update_function_update()

    def start(self):
        self.run(threaded=True)

    def stop(self, blocking=True):
        self._to_stop = True
        if blocking:
            while self._is_started:
                time.sleep(0.01)
                print('still started')
            print('now stopped')

    def is_started(self):
        return self._is_started

    def step(self):
        assert not self._is_started
        self.run(threaded=False, num_loops=1)
        return self.state

    @contextmanager
    def pause(self):
        self.stop(blocking=True)
        try:
            yield self
        finally:
            self.run(threaded=True)

    def update_attr(self, attr, type_):
        print('update_attr')
        setattr(self, attr, type_(getattr(self.state.simulator_state, attr)[0]))

    def update_space(self, box_size):
        print('update_space')
        self.displacement, self.shift = space.periodic(box_size)

    def update_function_update(self):
        print('update_function_update')
        self.init_fn, self.step_fn = self.dynamics_fn(self.displacement, self.shift, self.behavior_bank)

        def update_fn(_, state_and_neighbors):
            state, neighs = state_and_neighbors
            neighs = neighs.update(state.nve_state.position.center)
            return (self.step_fn(state=state, neighbor=neighs, agent_neighs_idx=self.agent_neighs_idx),
                    neighs)

        if self.to_jit:
            self.update_fn = jit(update_fn)
        else:
            self.update_fn = update_fn

    def init_state(self, state):
        print('init_state')
        self.state = self.init_fn(state, self.key)

    def update_neighbor_fn(self, box_size, neighbor_radius):
        print('update_neighbor_fn')
        self.neighbor_fn = partition.neighbor_list(self.displacement, box_size,
                                                   r_cutoff=neighbor_radius,
                                                   dr_threshold=10.,
                                                   capacity_multiplier=1.5,
                                                   # custom_mask_function=neigh_idx_mask,
                                                   format=partition.Sparse)

    def allocate_neighbors(self, position=None):
        print('allocate_neighbors')
        position = self.state.nve_state.position.center if position is None else position
        self.neighbors = self.neighbor_fn.allocate(position)
        mask = self.state.nve_state.entity_type[self.neighbors.idx[0]] == EntityType.AGENT.value
        self.agent_neighs_idx = self.neighbors.idx[:, mask]
        return self.neighbors

    def get_change_time(self):
        return 0

    def get_state(self):
        return self.state


if __name__ == "__main__":

    simulator_config = SimulatorConfig(to_jit=True)

    agent_configs = [AgentConfig(idx=i,
                                 x_position=np.random.rand() * simulator_config.box_size,
                                 y_position=np.random.rand() * simulator_config.box_size,
                                 orientation=np.random.rand() * 2. * np.pi)
                     for i in range(simulator_config.n_agents)]

    object_configs = [ObjectConfig(idx=simulator_config.n_agents + i,
                                   x_position=np.random.rand() * simulator_config.box_size,
                                   y_position=np.random.rand() * simulator_config.box_size,
                                   orientation=np.random.rand() * 2. * np.pi)
                      for i in range(simulator_config.n_objects)]

    state = converters.set_state_from_config_dict({StateType.AGENT: agent_configs,
                                                   StateType.OBJECT: object_configs,
                                                   StateType.SIMULATOR: [simulator_config]
                                                   })

    simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid)

    simulator.run(threaded=False, num_loops=10)


