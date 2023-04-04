import jax.numpy as jnp
from jax import jit
from jax import lax

import vivarium.simulator.behaviors as behaviors

from functools import partial

from vivarium.simulator.sim_computation import dynamics, Population

from vivarium.simulator import config

import time
import threading

import param


def sim_state_to_populations(sim_state, entity_slices):
    pop_dict = {}
    for e_type, e_slice in entity_slices.items():
        pop_dict[e_type] = Population(sim_state.positions[slice(*e_slice), :], sim_state.thetas[slice(*e_slice)], sim_state.entity_type[slice(*e_slice)][0])

    return pop_dict


class Simulator(param.Parameterized):
    simulation_config = param.ClassSelector(config.SimulatorConfig, instantiate=False)
    agent_config = param.ClassSelector(config.AgentConfig, instantiate=False)
    behavior_config = param.ClassSelector(config.BehaviorConfig, instantiate=False)
    population_config = param.ClassSelector(config.PopulationConfig, instantiate=False)
    is_started = param.Boolean(False)

    @param.depends('population_config', watch=True, on_init=True)
    def _update_state_neighbors(self):
        self.state = Population(positions=self.population_config.positions, thetas=self.population_config.thetas,
                                proxs=self.population_config.proxs, motors=self.population_config.motors,
                                entity_type=0)

        self.neighbors = self.simulation_config.neighbor_fn.allocate(self.population_config.positions)

    @param.depends('simulation_config.displacement', 'simulation_config.shift', 'simulation_config.map_dim',
                   'simulation_config.dt', 'agent_config.speed_mul', 'agent_config.theta_mul',
                   'agent_config.proxs_dist_max', 'agent_config.proxs_cos_min', 'agent_config.base_length',
                   'agent_config.wheel_diameter', 'behavior_config.entity_behaviors', 'behavior_config.behavior_bank',
                   watch=True, on_init=True)
    def _update_function_update(self):
        print("Update update function due to watched param dependencies")
        self.update_fn = dynamics(self.simulation_config, self.agent_config, self.behavior_config)

        if self.simulation_config.to_jit:
            self.update_fn = jit(self.update_fn)

    def set_behavior(self, e_idx, behavior_name):
        #self.behavior_config.behavior_bank[-e_idx - 1] = behaviors.apply_motors
        self.behavior_config.entity_behaviors = self.behavior_config.entity_behaviors.at[e_idx].set(self.behavior_config.behavior_name_map[behavior_name])
        #self.update_fn = dynamics(self.simulation_config, self.agent_config, self.behavior_config)

    def set_motors(self, e_idx, motors):
        if self.behavior_config.entity_behaviors[e_idx] != self.behavior_config.behavior_name_map['manual']:
            self.set_behavior(e_idx, 'manual')
        # self.behavior_config.entity_behaviors = self.behavior_config.entity_behaviors.at[e_idx].set(self.behavior_config.behavior_name_map['manual'])
        self.state = Population(positions=self.state.positions,
                                thetas=self.state.thetas,
                                proxs=self.state.proxs,
                                motors=self.state.motors.at[e_idx, :].set(jnp.array(motors)),
                                entity_type=self.state.entity_type)

    def run(self, threaded=False):
        if self.is_started:
            raise Exception("Simulator is already started")
        if threaded:
            threading.Thread(target=self._run).start()
        else:
            return self._run()

    def _run(self):

        self.is_started = True
        print('Run starts')
        while True:

            time.sleep(1. / self.simulation_config.freq)

            if not self.is_started:
                break
            if self.simulation_config.to_jit:
                new_state, neighbors = lax.fori_loop(0, self.simulation_config.num_steps_lax, self.update_fn,
                                                    (self.state, self.neighbors))
            else:
                #assert False, "not good, modifies self.state"
                #val = (self.state, self.neighbors)
                for i in range(0, self.simulation_config.num_steps_lax):
                    self.state, self.neighbors = self.update_fn(i, (self.state, self.neighbors))
                new_state = self.state
                #new_state, neighbors = val

            # If the neighbor list can't fit in the allocation, rebuild it but bigger.
            if neighbors.did_buffer_overflow:
                print('REBUILDING')
                neighbors = self.simulation_config.neighbor_fn.allocate(self.state.positions)
                new_state, neighbors = lax.fori_loop(0, self.simulation_config.num_lax_loops, self.update_fn, (self.state, neighbors))
                assert not neighbors.did_buffer_overflow

            self.state = new_state

        print('Run stops')

    def stop(self):
        self.is_started = False


if __name__ == "__main__":

    agent_config = config.AgentConfig()
    simulation_config = config.SimulatorConfig(agent_config=agent_config)
    population_config = config.PopulationConfig()
    behavior_config = config.BehaviorConfig(population_config=population_config)

    simulator = Simulator(simulation_config=simulation_config, agent_config=agent_config,
                          behavior_config=behavior_config, population_config=population_config)

    #simulator.set_motors(0, jnp.array([0., 0.]))
    simulator.run()
