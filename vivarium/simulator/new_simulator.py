import os
import time
import math
import pickle
import logging
import threading
import datetime

from functools import partial
from contextlib import contextmanager

import jax
import jax.numpy as jnp

from jax import jit
from jax import lax
from jax_md import space, partition, dataclasses
from flax import struct

from vivarium.controllers import converters
from vivarium.simulator.states import EntityType, AgentState, EntityState, ObjectState, StateType
# TODO :  # Import it later but atm defined in the same file
# from vivarium.simulator.states import State as SimState
from vivarium.environments.braitenberg.selective_sensing import State as EnvState


# TODO : Handle the simulator state, handle the weird jax_md functions ... 
from jax_md.dataclasses import dataclass
from jax_md import util, simulate, rigid_body

lg = logging.getLogger(__name__)

# TODO : just need to convert the simulator state and the elements inside en_state (not inside agents, objects ...)
@dataclass
class SimulatorState:
    idx: util.Array
    time: util.Array
    # ent_sub_types: util.Array # TODO : Maybe add it later
    box_size: util.Array
    max_agents: util.Array
    max_objects: util.Array
    num_steps_lax: util.Array
    dt: util.Array
    freq: util.Array
    neighbor_radius: util.Array
    to_jit: util.Array
    use_fori_loop: util.Array
    collision_alpha: util.Array
    collision_eps: util.Array

    # DONE : Added time
    @staticmethod
    def get_type(attr):
        if attr in ['idx', 'max_agents', 'max_objects', 'num_steps_lax']:
            return int
        elif attr in ['time', 'box_size', 'dt', 'freq', 'neighbor_radius', 'collision_alpha', 'collision_eps']:
            return float
        elif attr in ['to_jit', 'use_fori_loop']:
            return bool
        else:
            raise ValueError(f"Unknown attribute {attr}")
     
@dataclass
class SimState:
    simulator_state: SimulatorState
    entity_state: EntityState
    agent_state: AgentState
    object_state: ObjectState

    def field(self, stype_or_nested_fields):
        if isinstance(stype_or_nested_fields, StateType):
            name = stype_or_nested_fields.name.lower()
            nested_fields = (f'{name}_state', )
        else:
            nested_fields = stype_or_nested_fields

        res = self
        for f in nested_fields:
            res = getattr(res, f)

        return res

    def ent_idx(self, etype, entity_idx):
        return self.field(etype).ent_idx[entity_idx]

    def e_idx(self, etype):
        return self.entity_state.entity_idx[self.entity_state.entity_type == etype.value]

    def e_cond(self, etype):
        return self.entity_state.entity_type == etype.value

    def row_idx(self, field, ent_idx):
        return ent_idx if field == 'entity_state' else self.entity_state.entity_idx[jnp.array(ent_idx)]

    def __getattr__(self, name):
        def wrapper(e_type):
            value = getattr(self.entity_state, name)
            if isinstance(value, rigid_body.RigidBody):
                return rigid_body.RigidBody(center=value.center[self.e_cond(e_type)],
                                            orientation=value.orientation[self.e_cond(e_type)])
            else:
                return value[self.e_cond(e_type)]
        return wrapper

# TODO : Make that the state of simulator is the SimState (because will be used in CLient server communication)
# TODO : Create a property method that returns env state as self.sim_to_env_state(self.state)
class Simulator:
    def __init__(self, env, env_state, num_steps_lax=4, update_freq=-1, jit_step=True, use_fori_loop=True, seed=0):
        self.env = env

        self.key = jax.random.PRNGKey(seed)
        self.num_steps_lax = num_steps_lax
        self.freq = update_freq
        self.jit_step = jit_step
        self.use_fori_loop = use_fori_loop
        self.ent_sub_types = env_state.ent_sub_types # information about entities sub types in a dictionary, can't be given client side at the moment

        # transform the env state (only backend) into a jax md state with the older interface
        self.state = self.env_to_sim_state(env_state)

        # Attributes to start or stop the simulation
        self._is_started = False
        self._to_stop = False

        # Attributes to record simulation
        self.recording = False
        self.records = None
        self.saving_dir = None

        # TODO: Define which attributes are affected but these functions
        self.simulation_loop = self.select_simulation_loop_type(use_fori_loop)

    # Done : Add a partial so if args of simulator change it will be changed in this fn
    @partial(jax.jit, static_argnums=(0,))
    def env_to_sim_state(self, env_state):
        simulator_state = SimulatorState(
            idx=jnp.array([2]),
            time=jnp.array([env_state.time]),
            box_size=jnp.array([env_state.box_size]),
            max_agents=jnp.array([env_state.max_agents]),
            max_objects=jnp.array([env_state.max_objects]),
            dt=jnp.array([env_state.dt]),
            neighbor_radius=jnp.array([env_state.neighbor_radius]),
            collision_alpha=jnp.array([env_state.collision_alpha]),
            collision_eps=jnp.array([env_state.collision_eps]),
            num_steps_lax=jnp.array([self.num_steps_lax]),
            freq=jnp.array([self.freq]),
            use_fori_loop=jnp.array([self.use_fori_loop]),
            to_jit=jnp.array([self.jit_step])
        )

        sim_state = SimState(
            agent_state=env_state.agents,
            entity_state=env_state.entities,
            object_state=env_state.objects,
            simulator_state = simulator_state
        )

        return sim_state
        
    @partial(jax.jit, static_argnums=(0,))
    def sim_to_env_state(self, sim_state):
        sim = sim_state.simulator_state

        env_state = EnvState(
            time=sim.time[0],
            ent_sub_types=self.ent_sub_types,
            box_size=sim.box_size[0],
            max_agents=sim.max_agents[0],
            max_objects=sim.max_objects[0],
            dt=sim.dt[0],
            neighbor_radius=sim.neighbor_radius[0],
            collision_alpha=sim.collision_alpha[0],
            collision_eps=sim.collision_eps[0],
            entities=sim_state.entity_state,
            agents=sim_state.agent_state,
            objects=sim_state.object_state
        )

        return env_state
    
    # Add a method to directly get env state
    @property
    def env_state(self):
        return self.sim_to_env_state(self.state)


    # DONE : Remove num loops arg and removed 
    # TODO : Handle the num steps lax in environment side (not too hard to do)
    def _step(self, state, num_iterations):
        """Do a step in the simulation by applying the update function a few iterations on the state and the neighbors

        :param state: current simulation state 
        :param neighbors: current simulation neighbors array 
        :return: updated state and neighbors
        """
        # convert the sim_state into env state to call env.step()
        new_env_state = self.env.step(state=self.sim_to_env_state(state))

        # TODO : Remove this weird thing and just save the env state --> Because it is with this one that we can plot state in server side
        if self.recording:
            self.record(new_env_state)

        # return the next sim state
        return self.env_to_sim_state(new_env_state)
    

    def step(self):
        """Do a step in the simulation by calling _step"""
        self.state = self._step(self.state, self.num_steps_lax)
        return self.state

    # DONE : Added a return of the final state with run function
    def run(self, threaded=False, num_steps=math.inf, save=False, saving_name=None):
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
            _run = partial(self._run, num_steps=num_steps, save=save, saving_name=saving_name)
            threading.Thread(target=_run).start()
        else:
            self._run(num_steps=num_steps, save=save, saving_name=saving_name)
        return self.state


    def _run(self, num_steps, save, saving_name):
        """Function that runs the simulator for the desired number of steps. Used to be called either normally or in a thread.

        :param num_steps: number of simulation steps
        """
        # Encode that the simulation is started in the class
        self._is_started = True
        lg.info('Run starts')

        loop_count = 0
        sleep_time = 0

        if save:
            self.start_recording(saving_name)
    
        # Update the simulation with step for num_steps
        while loop_count < num_steps:
            start = time.time()
            if self._to_stop:
                self._to_stop = False
                break

            self.state = self._step(state=self.state, num_iterations=self.num_steps_lax)
            loop_count += 1

            # Sleep for updated sleep_time seconds
            end = time.time()
            sleep_time = self.update_sleep_time(frequency=self.freq, elapsed_time=end-start)
            time.sleep(sleep_time)

        if save:
            self.stop_recording()

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


    def select_simulation_loop_type(self, use_fori_loop):
        """Choose wether to use a lax or a classic simulation loop in function step

        :return: appropriate simulation loop
        """
        if use_fori_loop:
            return self.lax_simulation_loop
        else:
            return self.classic_simulation_loop
        

    def start_recording(self, saving_name):
        """Start the recording of the simulation
        :param saving_name: optional name of the saving file
        """
        if self.recording:
            lg.warning('Already recording')
        self.recording = True
        self.records = []

        # Either create a saving_dir with the given name or one with the current datetime
        if saving_name:
            saving_dir = f"Results/{saving_name}"
        else:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            saving_dir = f"Results/experiment_{current_time}"

        self.saving_dir = saving_dir
        # Create a saving dir if it doesn't exist yet, TODO : Add a warning if risk of overwritting already existing content
        os.makedirs(self.saving_dir, exist_ok=True)
        lg.info('Saving directory %s created', self.saving_dir)


    def record(self, data):
        """Record the desired data during a step
        :param data: saved data (e.g simulator.state)
        """
        if not self.recording:
            lg.warning('Recording not started yet.')
            return
        self.records.append(data)


    def save_records(self):
        """Save the recorded steps in a pickle file"""
        if not self.records:
            lg.warning('No records to save.')
            return

        saving_path = f"{self.saving_dir}/frames.pkl"
        with open(saving_path, 'wb') as f:
            pickle.dump(self.records, f)
            lg.info('Simulation frames saved in %s', saving_path)


    def stop_recording(self):
        """Stop the recording, save the recorded steps and reset recording information"""
        if not self.recording:
            lg.warning('Recording not started yet.')
            return

        self.save_records()
        self.recording = False
        self.records = []


    def load(self, saving_name):
        """Load data corresponding to saving_name 
        :param saving_name: name used while saving the data
        :return: loaded data
        """
        saving_path = f"Results/{saving_name}/frames.pkl"
        with open(saving_path, 'rb') as f:
            data = pickle.load(f)
            lg.info('Simulation loaded from %s', saving_path)
            return data
    

    # TODO : Add new attrbutes in this file when conversion functions are done
    def set_state(self, nested_field, ent_idx, column_idx, value):
        lg.info(f'set_state {nested_field} {ent_idx} {column_idx} {value}')
        row_idx = self.state.row_idx(nested_field[0], jnp.array(ent_idx))
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

    # DONE : No more update space in the simulator 
    # DONE : No more update fn in the simulator 
    # DONE : No more update neighbor functions in the simulator

    # Other functions
    # TODO : Remove ? 
    def get_change_time(self):
        return 0

    def get_state(self):
        return self.state
    
if __name__ == "__main__":
    # Test for init, run and convertion functions
    from vivarium.environments.braitenberg.selective_sensing import SelectiveSensorsEnv, init_state

    env_state = init_state()
    env = SelectiveSensorsEnv(state=env_state)

    simulator = Simulator(env=env, env_state=env_state)

    num_steps = 10
    sim_state = simulator.run(num_steps=num_steps)

    env_state = simulator.sim_to_env_state(sim_state)
    assert isinstance(env_state, EnvState)
    sim_state = simulator.env_to_sim_state(env_state)

    sim_state = simulator.run(num_steps=num_steps)
    assert isinstance(sim_state, SimState)