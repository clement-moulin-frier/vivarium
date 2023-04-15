import jax.numpy as jnp
from jax import jit, grad
from jax import lax
import jax
import numpy as np
from jax_md import space, partition, simulate, quantity, rigid_body

import vivarium.simulator.behaviors as behaviors

from functools import partial

from vivarium.simulator.sim_computation import dynamics, Population, total_collision_energy, RigidRobot, rigid_verlet_init_step, get_verlet_force_fn

from vivarium.simulator import config

import time
import threading
import math
import param
import requests

def sim_state_to_populations(sim_state, entity_slices):
    pop_dict = {}
    for e_type, e_slice in entity_slices.items():
        pop_dict[e_type] = Population(sim_state.positions[slice(*e_slice), :], sim_state.thetas[slice(*e_slice)], sim_state.entity_type[slice(*e_slice)][0])

    return pop_dict

def generate_positions_orientations(key, n_agents, box_size):
    key, subkey = jax.random.split(key)
    positions = box_size * jax.random.uniform(subkey, (n_agents, 2))
    key, subkey = jax.random.split(key)
    orientations = jax.random.uniform(subkey, (n_agents,), maxval=2 * np.pi)
    return positions, orientations

def generate_population(n_agents, box_size):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    positions, thetas = generate_positions_orientations(key, n_agents, box_size)
    proxs = jnp.zeros((n_agents, 2))
    motors = jnp.zeros((n_agents, 2))
    return Population(position=positions, theta=thetas, prox=proxs, motor=motors, entity_type=0)


class Simulator():
    # simulation_config = param.ClassSelector(config.SimulatorConfig, instantiate=False)
    # agent_config = param.ClassSelector(config.AgentConfig, instantiate=False)
    # # behavior_config = param.ClassSelector(config.BehaviorConfig, instantiate=False)
    # # population_config = param.ClassSelector(config.PopulationConfig, instantiate=False)
    # engine_config = param.ClassSelector(config.EngineConfig, instantiate=False)
    # is_started = param.Boolean(False)
    # # engine_config = param.ClassSelector(config.EngineConfig)

    def __init__(self, simulation_config, agent_config, engine_config):
        # super().__init__(**params)
        self.simulation_config = simulation_config
        self.agent_config = agent_config
        self.engine_config = engine_config
        self.engine_config.behavior_name_map['manual'] = len(self.engine_config.behavior_bank) - 1
        self.is_started = False
        # self.simulation_config.param.watch_values(self._record_change, self.simulation_config.export_fields, queued=True)
        # self._recorded_change_dict = {}

    # def _record_change(self, **kwargs):
    #     self._recorded_change_dict.update(kwargs)
    #
    # def get_recorded_changes(self):
    #     d = dict(self._recorded_change_dict)
    #     self._recorded_change_dict = {}
    #     return d
    #
    # @param.depends('simulation_config.displacement', 'simulation_config.box_size', 'agent_config.neighbor_radius',
    #                watch=True, on_init=True)

    @property
    def state(self):
        return self._state

    def init_simulator(self):
        self.update_neighbor_fn()
        self.update_state_neighbors()
        self.update_function_update()
    def update_neighbor_fn(self):
        print('_update_neighbor_fn')
        self.neighbor_fn = partition.neighbor_list(self.simulation_config.displacement,
                                                   self.simulation_config.box_size,
                                                   r_cutoff=self.agent_config.neighbor_radius,
                                                   dr_threshold=10.,
                                                   capacity_multiplier=1.5,
                                                   format=partition.Sparse)

    # @param.depends('simulation_config.n_agents', 'simulation_config.box_size', watch=True, on_init=True)
    def update_state_neighbors(self):
        print('_update_state_neighbors')
        # self.is_started = False
        self._state = generate_population(self.simulation_config.n_agents, self.simulation_config.box_size)
        # self._state = Population(positions=self.population_config.positions, thetas=self.population_config.thetas,
        #                         proxs=self.population_config.proxs, motors=self.population_config.motors,
        #                         entity_type=0)

        self.neighbors = self.neighbor_fn.allocate(self._state.position)
        # self.run(threaded=True)‹‹

    # @param.depends('simulation_config.displacement', 'simulation_config.shift', 'simulation_config.map_dim',
    #                'simulation_config.dt', 'agent_config.speed_mul', 'agent_config.theta_mul',
    #                'agent_config.proxs_dist_max', 'agent_config.proxs_cos_min', 'agent_config.base_length',
    #                'agent_config.wheel_diameter', 'simulation_config.entity_behaviors', 'engine_config.behavior_bank',
    #                watch=True, on_init=True)
    def update_function_update(self):
        print("_update_function_update")
        self.update_fn = dynamics(self.engine_config, self.simulation_config, self.agent_config)

        if self.simulation_config.to_jit:
            self.update_fn = jit(self.update_fn)

    # @param.depends('simulation_config.n_agents', watch=True, on_init=True)
    def update_behaviors(self):
        print('_update_behaviors')
        print("WARNING: why all zeros?")
        self.simulation_config.entity_behaviors = np.zeros(self.simulation_config.n_agents, dtype=int)

    def set_behavior(self, e_idx, behavior_name):
        self.simulation_config.entity_behaviors[e_idx] = self.engine_config.behavior_name_map[behavior_name]  # self.behavior_config.entity_behaviors.at[e_idx].set(self.behavior_config.behavior_name_map[behavior_name])

    def set_motors(self, e_idx, motors):
        if self.behavior_config.entity_behaviors[e_idx] != self.behavior_config.behavior_name_map['manual']:
            self.set_behavior(e_idx, 'manual')
        self._state = Population(positions=self._state.positions,
                                thetas=self._state.thetas,
                                proxs=self._state.proxs,
                                motors=self._state.motors.at[e_idx, :].set(jnp.array(motors)),
                                entity_type=self._state.entity_type)

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
            # print(self.simulation_config.entity_behaviors)
            if self.simulation_config.freq is not None:
                time.sleep(1. / self.simulation_config.freq)

            if not self.is_started:
                break
            if self.simulation_config.use_fori_loop:
                new_state, neighbors = lax.fori_loop(0, self.simulation_config.num_steps_lax, self.update_fn,
                                                    (self._state, self.neighbors))
            else:
                #assert False, "not good, modifies self._state"
                #val = (self._state, self.neighbors)
                new_state = self._state
                neighbors = self.neighbors
                for i in range(0, self.simulation_config.num_steps_lax):
                    new_state, neighbors = self.update_fn(i, (new_state, neighbors))
                #new_state = self._state
                #new_state, neighbors = val

            # If the neighbor list can't fit in the allocation, rebuild it but bigger.
            if neighbors.did_buffer_overflow:
                print('REBUILDING')
                neighbors = self.simulation_config.neighbor_fn.allocate(self._state.positions)
                new_state, neighbors = lax.fori_loop(0, self.simulation_config.num_lax_loops, self.update_fn, (self._state, neighbors))
                assert not neighbors.did_buffer_overflow

            self._state = new_state
            self.neighbors = neighbors

            loop_count += 1

        print('Run stops')

    def stop(self):
        self.is_started = False



class VerletSimulator(Simulator):
    def __init__(self, simulation_config, agent_config, engine_config):
        super().__init__(simulation_config, agent_config, engine_config)
        #self.init_simulator()
        self._shape = rigid_body.monomer

    def update_neighbor_fn(self):
        self._energy_fn = get_verlet_force_fn(self.engine_config, self.simulation_config, self.agent_config)  # partial(total_collision_energy, base_length=self.agent_config.base_length, displacement=self.simulation_config.displacement)
        self.neighbor_fn = partition.neighbor_list(self.simulation_config.displacement,
                                                   self.simulation_config.box_size,
                                                   r_cutoff=self.agent_config.neighbor_radius,
                                                   dr_threshold=10.,
                                                   capacity_multiplier=1.5,
                                                   format=partition.Sparse)
        #self.neighbor_fn, self._energy_fn = rigid_body.point_energy_neighbor_list(energy_fn, neighbor_fn, shape=self._shape)

    def update_state_neighbors(self):
        # pop = generate_population(self.simulation_config.n_agents, self.simulation_config.box_size)
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        # state = simulate.NVEState(position=pop.position, momentum=None, force=1., mass=1.)
        # state = simulate.canonicalize_mass(state)
        # kT = 0
        pop = generate_population(self.simulation_config.n_agents, self.simulation_config.box_size)
        bodies = RigidRobot(center=pop.position, orientation=pop.theta,
                            prox=jnp.zeros((self.simulation_config.n_agents, 2)),
                            motor=jnp.zeros((self.simulation_config.n_agents, 2)),
                            entity_type=0)
        #self.neighbors = self.neighbor_fn.allocate(bodies.center)

        self.neighbors = self.neighbor_fn.allocate(bodies.center)
        self._init_fn, self._step_fn = rigid_verlet_init_step(self._energy_fn, self.simulation_config.shift, self.simulation_config.dt)
        self._state = self._init_fn(key, bodies.to_rigid_body(), mass=self._shape.mass(),
                                    neighbor=self.neighbors) #simulate.initialize_momenta(state, key, kT)
    @property
    def state(self):

        state = Population(position=self._state.position.center,
                           theta=self._state.position.orientation,
                           prox=jnp.zeros((self.simulation_config.n_agents, 2)),
                           motor=jnp.zeros((self.simulation_config.n_agents, 2)),
                           entity_type=0)
        return state
    def update_function_update(self):
        # step_fn = partial(simulate.velocity_verlet, shift_fn=self.simulation_config.shift, dt=self.simulation_config.dt)
        def update_fn(_, state_and_neighbors):
            state, neighs = state_and_neighbors
            neighs = neighs.update(state.position.center)
            #print('upd', self._force_fn(self._state.position, neighs))
            return (self._step_fn(state=state, neighbor=neighs),
                    neighs)

        print("_update_function_update")
        self.update_fn = update_fn

        if self.simulation_config.to_jit:
            self.update_fn = jit(self.update_fn)

    # # @param.depends('simulation_config.n_agents', watch=True, on_init=True)
    # def update_behaviors(self):
    #     print('_update_behaviors')
    #     self.simulation_config.entity_behaviors = np.zeros(self.simulation_config.n_agents, dtype=int)
    #
    # def set_behavior(self, e_idx, behavior_name):
    #     self.simulation_config.entity_behaviors[e_idx] = self.engine_config.behavior_name_map[behavior_name]  # self.behavior_config.entity_behaviors.at[e_idx].set(self.behavior_config.behavior_name_map[behavior_name])
    #
    # def set_motors(self, e_idx, motors):
    #     if self.behavior_config.entity_behaviors[e_idx] != self.behavior_config.behavior_name_map['manual']:
    #         self.set_behavior(e_idx, 'manual')
    #     self._state = Population(positions=self._state.positions,
    #                             thetas=self._state.thetas,
    #                             proxs=self._state.proxs,
    #                             motors=self._state.motors.at[e_idx, :].set(jnp.array(motors)),
    #                             entity_type=self._state.entity_type)


if __name__ == "__main__":

    agent_config = config.AgentConfig()
    simulation_config = config.SimulatorConfig()
    simulation_config.to_jit = False
    engine_config = config.EngineConfig()

    simulator = VerletSimulator(simulation_config=simulation_config, agent_config=agent_config,
                          engine_config=engine_config)

    simulator.init_simulator()
    #simulator.set_motors(0, jnp.array([0., 0.]))
    # simulator.is_started = True
    simulator.run()

