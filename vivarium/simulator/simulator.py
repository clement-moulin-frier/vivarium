import jax.numpy as jnp
from jax import jit, grad
from jax import lax
import jax
import numpy as np
from jax_md import space, partition, simulate, quantity, rigid_body

import vivarium.simulator.behaviors as behaviors

from functools import partial
from collections import namedtuple


from vivarium.simulator.sim_computation import dynamics, Population, total_collision_energy, RigidRobot, dynamics_rigid, get_verlet_force_fn, NVEState

from vivarium.simulator.config import AgentConfig, SimulatorConfig

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
    return key, positions, orientations

def generate_population(n_agents, box_size):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    positions, thetas = generate_positions_orientations(key, n_agents, box_size)
    proxs = jnp.zeros((n_agents, 2))
    motors = jnp.zeros((n_agents, 2))
    return Population(position=positions, theta=thetas, prox=proxs, motor=motors, entity_type=0)


class EngineConfig(param.Parameterized):
    simulation_config = param.ClassSelector(SimulatorConfig, instantiate=False)
    n_agents = param.Integer(10)
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
        self.agent_configs = self.agent_configs or [AgentConfig() for _ in range(self.n_agents)]
        self.key = key = jax.random.PRNGKey(0)
        # self.agent_config = self.simulation_config.agent_configs[0]  # Temporary
        # self.behavior_name_map = {beh.name: i for i, beh in enumerate(behaviors.linear_behavior_enum)}
        # self.behavior_name_map['manual'] = len(self.behavior_bank) - 1
        # self.entity_behaviors = self.simulation_config.entity_behaviors()  # self.simulation_config.entity_behaviors or 2 * np.ones(self.simulation_config.n_agents, dtype=int)
        # self.displacement, self.shift = space.periodic(self.box_size)
        self.simulation_config.param.watch(self.update_space, ['box_size'], onlychanged=True, precedence=0)
        self.param.watch(self.update_neighbor_fn, ['displacement'], onlychanged=True, precedence=1)
        self.simulation_config.param.watch(self.update_state, ['box_size'], onlychanged=True)
        self.simulation_config.param.watch(self.update_neighbor_fn, ['neighbor_radius'], onlychanged=True)
        # self.simulation_config.param.watch(self.update_neighbors, ['n_agents'], onlychanged=True, precedence=3)
        self.param.watch(self.update_neighbors, ['neighbor_fn'], onlychanged=True, precedence=3)
        self.param.watch(self.update_function_update, ['displacement', 'shift', 'behavior_bank'], onlychanged=True, precedence=2)
        self.simulation_config.param.watch(self.update_function_update,
                                           ['dt', 'map_dim', 'to_jit'],
                                           onlychanged=True)
        # for config in self.simulation_config.agent_configs:
        #     config.param.watch(self.update_state, ['behavior'], onlychanged=True, precedence=1)
        # self.agent_config.param.trigger('behavior')
        # with param.parameterized.batch_call_watchers(self):
        # self.simulation_config.param.trigger('box_size')
        for config in self.agent_configs:
            # config.param.watch(self.update_state, ['behavior'], onlychanged=True, precedence=1)
            # config.param.watch(self.update_neighbor_fn, ['neighbor_radius'], onlychanged=True)
            # config.param.watch(self.update_function_update,
            #              ['proxs_dist_max', 'proxs_cos_min', 'base_length', 'wheel_diameter', 'behavior', 'speed_mul', 'theta_mul'],
            #              onlychanged=True, precedence=3)
            config.param.watch(self.update_state, list(config.to_dict().keys()), onlychanged=True)

        # self.param.watch(self.update_state, ['simulation_config.agent_configs'], onlychanged=True) # list(config.to_dict().keys())

        MockEvent = namedtuple('MockEvent', ['name', 'new'])
        with param.parameterized.discard_events(self):
            self.update_space(MockEvent(name='box_size', new=self.simulation_config.box_size))
            self.update_function_update()
            self.update_state()
            self.update_neighbor_fn()
            self.update_neighbors()

    def update_space(self, event):
        print('update_space', event.name)
        # with param.parameterized.batch_call_watchers(self):
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
        # for e in events:
        #     if e.name == 'n_agents' and self.state is not None:
        #         if e.new < e.new:

        if len(events) == 0:  # kind of hack: when we initialize the state for the first time
            self.key, subkey = jax.random.split(self.key)
            self.key, positions, orientations = generate_positions_orientations(key=self.key,
                                                                      n_agents=len(self.agent_configs),
                                                                      box_size=self.simulation_config.box_size)

            # for e in events:
            #     if e.name == "n_agents":
            #         if e.new < e.old:
            #             self.simulation_config.agent_configs = self.simulation_config.agent_configs[:e.new]
            #         elif e.new > e.old:
            #             self.simulation_config.agent_configs += [config.AgentConfig() for _ in range(e.new - e.old)]
            self.state = self.init_fn(self.key, positions=positions, orientations=orientations,
                                      agent_configs_as_array_dict=self.agent_configs_as_array_dict())
        else:  # when a change is made from the interface or controller
            self.state = self.state.set(**self.agent_configs_as_array_dict())
    # @param.depends('simulation_config.n_agents', 'simulation_config.box_size', watch=True, on_init=True)
    def update_neighbors(self, *events):
        if self.state is None:
            return
        print('update_neighbors',  [e.name for e in events])
        for e in events:
            if e.name == 'n_agents':
                assert self.state.position.center.shape[0] == e.new

        self.neighbors = self.neighbor_fn.allocate(self.state.position.center)

        # self._state = Population(positions=self.population_config.positions, thetas=self.population_config.thetas,
        #                         proxs=self.population_config.proxs, motors=self.population_config.motors,
        #                         entity_type=0)

        # self.run(threaded=True)‹‹

    # @param.depends('simulation_config.displacement', 'simulation_config.shift', 'simulation_config.map_dim',
    #                'simulation_config.dt', 'agent_config.speed_mul', 'agent_config.theta_mul',
    #                'agent_config.proxs_dist_max', 'agent_config.proxs_cos_min', 'agent_config.base_length',
    #                'agent_config.wheel_diameter', 'simulation_config.entity_behaviors', 'engine_config.behavior_bank',
    #                watch=True, on_init=True)
    def update_function_update(self, *events):
        print("_update_function_update", [e.name for e in events])
        self.init_fn, self.step_fn = self.dynamics_fn(self.displacement, self.shift,
                                                      self.simulation_config.map_dim, self.simulation_config.dt,
                                                      self.behavior_bank)

        def update_fn(_, state_and_neighbors):
            state, neighs = state_and_neighbors
            neighs = neighs.update(state.position.center)
            #print('upd', self._force_fn(self._state.position, neighs))
            return (self.step_fn(state=state, neighbor=neighs),
                    neighs)

        if self.simulation_config.to_jit:
            self.update_fn = jit(update_fn)
        else:
            self.update_fn = update_fn

        # if self.simulation_config.to_jit:  # Now in VerletSimulator, but better here?
        #     self.step_fn = jit(self.step_fn)
        #     print('jitted')

    # @param.depends('simulation_config.n_agents', watch=True, on_init=True)
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


class Simulator():
    # simulation_config = param.ClassSelector(config.SimulatorConfig, instantiate=False)
    # agent_config = param.ClassSelector(config.AgentConfig, instantiate=False)
    # # behavior_config = param.ClassSelector(config.BehaviorConfig, instantiate=False)
    # # population_config = param.ClassSelector(config.PopulationConfig, instantiate=False)
    # engine_config = param.ClassSelector(config.EngineConfig, instantiate=False)
    # is_started = param.Boolean(False)
    # # engine_config = param.ClassSelector(config.EngineConfig)

    def __init__(self, engine_config):
        # super().__init__(**params)
        # self.simulation_config = simulation_config
        # self.agent_config = simulation_config.agent_configs[0]
        self.engine_config = engine_config
        self.simulation_config = self.engine_config.simulation_config
        self.agent_configs = self.engine_config.agent_configs
        self._state = self.engine_config.state
        self.neighbors = self.engine_config.neighbors
        # self.behavior_bank = [partial(behaviors.linear_behavior,
        #                               matrix=behaviors.linear_behavior_matrices[beh])
        #                       for beh in behaviors.linear_behavior_enum] + [behaviors.apply_motors]
        # self.behavior_name_map = {beh.name: i for i, beh in enumerate(behaviors.linear_behavior_enum)}
        # self.behavior_name_map['manual'] = len(self.simulation_config.behavior_bank) - 1
        # self.entity_behaviors = self.simulation_config.entity_behaviors()  # self.simulation_config.entity_behaviors or 2 * np.ones(self.simulation_config.n_agents, dtype=int)
        # self.displacement, self.shift = space.periodic(self.box_size)
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
        return self.engine_config.state

    def init_simulator(self):
        self.update_neighbor_fn()
        self.update_state_neighbors()
        self.update_function_update()

    def _update_ds(self, event):
        print('_update_ds')
        self.displacement, self.shift = space.periodic(event.new)
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
        print("update_function_update")
        self.update_fn = dynamics(self.simulation_config)

        if self.simulation_config.to_jit:
            self.update_fn = jit(self.update_fn)

    # @param.depends('simulation_config.n_agents', watch=True, on_init=True)
    def update_behaviors(self):
        print('_update_behaviors')
        self.simulation_config.entity_behaviors = np.zeros(self.simulation_config.n_agents, dtype=int)

    def set_behavior(self, e_idx, behavior_name):
        self.simulation_config.entity_behaviors[e_idx] = self.simulation_config.behavior_name_map[behavior_name]  # self.behavior_config.entity_behaviors.at[e_idx].set(self.behavior_config.behavior_name_map[behavior_name])

    def set_motors(self, e_idx, motors):
        #if self.behavior_config.entity_behaviors[e_idx] != self.behavior_config.behavior_name_map['manual']:
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
                new_state = self.engine_config.state
                neighbors = self.engine_config.neighbors
                for i in range(0, self.simulation_config.num_steps_lax):
                    new_state, neighbors = self.engine_config.update_fn(i, (new_state, neighbors))
                #new_state = self._state
                #new_state, neighbors = val
            # If the neighbor list can't fit in the allocation, rebuild it but bigger.
            if neighbors.did_buffer_overflow:
                print('REBUILDING')
                neighbors = self.simulation_config.neighbor_fn.allocate(self._state.positions)
                new_state, neighbors = lax.fori_loop(0, self.simulation_config.num_lax_loops, self.update_fn, (self._state, neighbors))
                assert not neighbors.did_buffer_overflow
            self.engine_config.state = new_state
            self.engine_config.neighbors = neighbors
            # print(loop_count, self._state.position.center[:, 0])

            loop_count += 1

        print('Run stops')

    def stop(self):
        self.is_started = False



class VerletSimulator(Simulator):
    def __init__(self, engine_config):
        super().__init__(engine_config)

        # Should the code below be in EngineConfig instead?

        #self.init_simulator()
        # self._shape = rigid_body.monomer

    def update_neighbor_fn(self):
        self._energy_fn = get_verlet_force_fn(self.simulation_config, self.agent_config)  # partial(total_collision_energy, base_length=self.agent_config.base_length, displacement=self.simulation_config.displacement)
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
        self._init_fn, self._step_fn = dynamics_rigid(self._energy_fn, self.simulation_config)
        self._state = self._init_fn(key, bodies, mass=self._shape.mass(),
                                    neighbor=self.neighbors) #simulate.initialize_momenta(state, key, kT)
    # @property
    # def state(self):
    #
    #     state = Population(position=self._state.position.center,
    #                        theta=self._state.position.orientation,
    #                        prox=self._state.prox,
    #                        motor=self._state.motor,
    #                        entity_type=0)
    #     return state
    def update_function_update(self):
        # step_fn = partial(simulate.velocity_verlet, shift_fn=self.simulation_config.shift, dt=self.simulation_config.dt)
        _, self._step_fn = dynamics_rigid(self._energy_fn, self.simulation_config, self.agent_config)
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

    def set_motors(self, e_idx, motors):
        #if self.behavior_config.entity_behaviors[e_idx] != self.behavior_config.behavior_name_map['manual']:
        self.set_behavior(e_idx, 'manual')
        motor = self._state.motor.at[e_idx, :].set(jnp.array(motors))
        self._state = self._state.set(motor=motor)
        # self._state = NVEState(position=self._state.position, momentum=self._state.momentum,
        #                        force=self._state.force, mass=self._state.mass,
        #                        prox=self._state.prox, motor=motor, entity_type=self._state.entity_type)


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


    # agent_config = config.AgentConfig()
    simulation_config = SimulatorConfig(to_jit=True)

    engine_config = EngineConfig(simulation_config=simulation_config, dynamics_fn=dynamics_rigid)
    # engine_config.step_fn(state=engine_config.state, neighbor=engine_config.neighbors)

    # print('n_agents changes')
    # simulation_config.n_agents = 5
    # engine_config.step_fn(state=engine_config.state, neighbor=engine_config.neighbors)
    #
    # print('change')
    # simulation_config.agent_configs[4].wheel_diameter = 21.

    simulator = VerletSimulator(engine_config=engine_config)

    # simulator.init_simulator()
    #simulator.set_motors(0, jnp.array([0., 0.]))
    # simulator.is_started = True

    simulator.run()

