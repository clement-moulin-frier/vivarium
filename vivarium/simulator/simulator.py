import time
import threading
import math
import logging

from functools import partial
from contextlib import contextmanager

import jax
import jax.numpy as jnp

from jax import jit
from jax import lax
from jax_md import space, partition, dataclasses

from vivarium.simulator.sim_computation import EntityType, SimulatorState
from vivarium.controllers import converters

lg = logging.getLogger(__name__)


class Simulator:
    def __init__(self, state, behavior_bank, dynamics_fn):

        self.state = state
        self.behavior_bank = behavior_bank
        self.dynamics_fn = dynamics_fn

        # TODO: explicitely copy the attributes of simulator_state (prevents linting errors and easier to understand which element is an attriute of the class)
        all_attrs = [f.name for f in dataclasses.fields(SimulatorState)]
        for attr in all_attrs:
            self.update_attr(attr, SimulatorState.get_type(attr))

        self._is_started = False
        self._to_stop = False
        self.key = jax.random.PRNGKey(0)

        # TODO: Define which attributes are affected but these functions
        self.update_space(self.box_size)
        self.update_function_update()
        self.init_state(state)
        self.update_neighbor_fn(self.box_size, self.neighbor_radius)
        self.allocate_neighbors()
        self.simulation_loop = self.select_simulation_loop_type()


    def classic_simulation_loop(self, state, neighbors, num_iterations):
        """Update the state and the neighbors on a few iterations with a classic python loop

        :param state: current_state of the simulation
        :param neighbors: array of neighbors for simulation entities
        :return: state, neighbors
        """
        for i in range(0, num_iterations):
            state, neighbors = self.update_fn(i, (state, neighbors))
        return state, neighbors

    def lax_simulation_loop(self, state, neighbors, num_iterations):
        """Update the state and the neighbors on a few iterations with lax loop

        :param state: current_state of the simulation
        :param neighbors: array of neighbors for simulation entities
        :return: state, neighbors
        """
        state, neighbors = lax.fori_loop(0, num_iterations, self.update_fn, (state, neighbors))
        return state, neighbors

    def select_simulation_loop_type(self):
        """Choose wether to use a lax or a classic simulation loop in function step

        :return: appropriate simulation loop
        """
        if self.state.simulator_state.use_fori_loop:
            return self.lax_simulation_loop
        else:
            return self.classic_simulation_loop
    
    def step(self, state, neighbors):
        """Do a step in the simulation by applying the update function a few iterations on the state and the neighbors

        :param state: current simulation state 
        :param neighbors: current simulation neighbors array 
        :return: updated state and neighbors
        """
        # Create a copy of the current state in case of neighbor buffer overflow
        current_state = state
        # TODO : find a more explicit name than num_steps_lax and modify it in all the pipeline
        new_state, neighbors = self.simulation_loop(state=current_state, neighbors=neighbors, num_iterations=self.num_steps_lax)

        # If the neighbor list can't fit in the allocation, rebuild it but bigger.
        if neighbors.did_buffer_overflow:
            lg.warning('REBUILDING NEIGHBORS ARRAY')
            neighbors = self.allocate_neighbors(current_state.nve_state.position.center)
            # Because there was an error, we need to re-run this simulation loop from the copy of the current_state we created
            new_state, neighbors = self.simulation_loop(state=current_state, neighbors=neighbors, num_iterations=self.num_steps_lax)
            # Check that neighbors array is now ok but should be the case (allocate neighbors tries to compute a new list that is large enough according to the simulation state)
            assert not neighbors.did_buffer_overflow

        return new_state, neighbors

    def run(self, threaded=False, num_steps=math.inf):
        """Run the simulator for the desired number of timesteps, either in a separate thread or not 

        :param threaded: wether to run the simulation in a thread or not, defaults to False
        :param num_steps: number of step loops before stopping the simulation run, defaults to math.inf
        :raises ValueError: raise an error if the simulator is already running 
        """
        # Check is the simulator isn't already running
        if self._is_started:
            raise ValueError("Simulator is already started")
        # Else run it either in a thread or not
        if threaded:
            # Set the num_loops attribute with a partial func to launch _run in a thread
            _run = partial(self._run, num_steps=num_steps)
            threading.Thread(target=_run).start()
        else:
            self._run(num_steps)

    def _run(self, num_steps):
        """Function that runs the simulator for the desired number of steps. Used to be called either normally or in a thread.

        :param num_steps: number of simulation steps
        """
        # Encode that the simulation is started in the class
        self._is_started = True
        lg.info('Run starts')

        loop_count = 0
        sleep_time = 0
    
        # Update the simulation with step for num_steps
        while loop_count < num_steps:
            start = time.time()
            if self._to_stop:
                self._to_stop = False
                break

            self.state, self.neighbors = self.step(state=self.state, neighbors=self.neighbors)
            loop_count += 1

            # Sleep for updated sleep_time seconds
            end = time.time()
            sleep_time = self.update_sleep_time(frequency=self.freq, elapsed_time=end-start)
            time.sleep(sleep_time)

        # Encode that the simulation isn't started anymore 
        self._is_started = False
        lg.info('Run stops')

    def update_sleep_time(self, frequency, elapsed_time):
        """Compute the time we need to sleep to respect the update frequency

        :param frequency: update state frequency
        :param elapsed_time: time already used to compute the state
        :return: time needed to sleep in addition to elapsed time to respect the frequency 
        """
        # if we use the freq, compute the correct sleep time
        if float(frequency) > 0.:
            perfect_time = 1. / float(frequency)
            sleep_time = max(perfect_time - elapsed_time, 0)
        # Else set it to zero
        else:
            sleep_time = 0
        return sleep_time

    def set_state(self, nested_field, nve_idx, column_idx, value):
        lg.info(f'set_state {nested_field} {nve_idx} {column_idx} {value}')
        row_idx = self.state.row_idx(nested_field[0], jnp.array(nve_idx))
        col_idx = None if column_idx is None else jnp.array(column_idx)
        change = converters.rec_set_dataclass(self.state, nested_field, row_idx, col_idx, value)
        self.state = self.state.set(**change)

        if nested_field[0] == 'simulator_state':
            self.update_attr(nested_field[1], SimulatorState.get_type(nested_field[1]))

        if nested_field == ('simulator_state', 'box_size'):
            self.update_space(self.box_size)

        if nested_field in (('simulator_state', 'box_size'), ('simulator_state', 'neighbor_radius')):
            self.update_neighbor_fn(box_size=self.box_size, neighbor_radius=self.neighbor_radius)

        if nested_field in (('simulator_state', 'box_size'), ('simulator_state', 'dt'), ('simulator_state', 'to_jit')):
            self.update_function_update()


    # Functions to start, stop, pause

    def start(self):
        self.run(threaded=True)

    def stop(self, blocking=True):
        self._to_stop = True
        if blocking:
            while self._is_started:
                time.sleep(0.01)
                lg.info('still started')
            lg.info('now stopped')

    def is_started(self):
        return self._is_started

    @contextmanager
    def pause(self):
        self.stop(blocking=True)
        try:
            yield self
        finally:
            self.run(threaded=True)


    # Other update functions

    def update_attr(self, attr, type_):
        lg.info('update_attr')
        setattr(self, attr, type_(getattr(self.state.simulator_state, attr)[0]))

    def update_space(self, box_size):
        lg.info('update_space')
        self.displacement, self.shift = space.periodic(box_size)

    def update_function_update(self):
        lg.info('update_function_update')
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
        lg.info('init_state')
        self.state = self.init_fn(state, self.key)


    # Neighbor functions

    def update_neighbor_fn(self, box_size, neighbor_radius):
        lg.info('update_neighbor_fn')
        self.neighbor_fn = partition.neighbor_list(self.displacement, box_size,
                                                   r_cutoff=neighbor_radius,
                                                   dr_threshold=10.,
                                                   capacity_multiplier=1.5,
                                                   # custom_mask_function=neigh_idx_mask,
                                                   format=partition.Sparse)

    def allocate_neighbors(self, position=None):
        lg.info('allocate_neighbors')
        position = self.state.nve_state.position.center if position is None else position
        self.neighbors = self.neighbor_fn.allocate(position)
        mask = self.state.nve_state.entity_type[self.neighbors.idx[0]] == EntityType.AGENT.value
        self.agent_neighs_idx = self.neighbors.idx[:, mask]
        return self.neighbors
    

    # Other functions

    def get_change_time(self):
        return 0

    def get_state(self):
        return self.state
