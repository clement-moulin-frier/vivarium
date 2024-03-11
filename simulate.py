import argparse

import numpy as np
import jax.numpy as jnp

from vivarium.simulator.sim_computation import dynamics_rigid, StateType
from vivarium.controllers.config import AgentConfig, ObjectConfig, SimulatorConfig
from vivarium.controllers import converters
from vivarium.simulator.simulator import Simulator
import vivarium.simulator.behaviors as behaviors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--col_eps', type=float, required=False, default=3.)
    parser.add_argument('--col_alpha', type=float, required=False, default=0.3)
    parser.add_argument('--col_dist_mul', type=float, required=False, default=6)
    args = parser.parse_args()

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

    col_params = {'eps': args.col_eps, 'alpha': args.col_alpha, 'dist_mul': args.col_dist_mul}
    simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid, col_params)

    simulator.run(threaded=False, num_loops=10)

