import jax.numpy as jnp
from jax import jit
from jax import lax

import vivarium.simulator.behaviors as behaviors

from functools import partial

from vivarium.simulator.sim_computation import dynamics, Population

from vivarium.simulator import config

import time
import threading


def sim_state_to_populations(sim_state, entity_slices):
    pop_dict = {}
    for e_type, e_slice in entity_slices.items():
        pop_dict[e_type] = Population(sim_state.positions[slice(*e_slice), :], sim_state.thetas[slice(*e_slice)], sim_state.entity_type[slice(*e_slice)][0])

    return pop_dict


class Simulator():
    def __init__(self, simulation_config, agent_config, behavior_config, population_config):

        self.simulation_config = simulation_config
        self.agent_config = agent_config
        self.behavior_config = behavior_config
        self.population_config = population_config

        self.state = Population(positions=population_config.positions, thetas=population_config.thetas,
                                proxs=population_config.proxs, motors=population_config.motors,
                                entity_type=0)

        self.neighbors = self.simulation_config.neighbor_fn.allocate(population_config.positions)

        self.update_fn = dynamics(simulation_config, agent_config, behavior_config)

        if self.simulation_config.to_jit:
            self.update_fn = jit(self.update_fn)

        self.is_started = False

    def set_behavior(self, e_idx, behavior_name):
        #self.behavior_config.behavior_bank[-e_idx - 1] = behaviors.apply_motors
        self.behavior_config.entity_behaviors = self.behavior_config.entity_behaviors.at[e_idx].set(self.behavior_config.behavior_name_map[behavior_name])
        self.update_fn = dynamics(self.simulation_config, self.agent_config, self.behavior_config)

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

    simulator = Simulator(simulation_config, agent_config, behavior_config, population_config)

    simulator.set_motors(0, jnp.array([0., 0.]))
    simulator.run()
