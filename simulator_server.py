import logging
import argparse

import numpy as np

import vivarium.simulator.behaviors as behaviors
from vivarium.controllers.config import SimulatorConfig, AgentConfig, ObjectConfig
from vivarium.simulator.sim_computation import StateType
from vivarium.simulator.simulator import Simulator
from vivarium.simulator.sim_computation import dynamics_rigid
from vivarium.controllers.converters import set_state_from_config_dict
from vivarium.simulator.grpc_server.simulator_server import serve


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--col_eps', type=float, required=False, default=0.3)
    parser.add_argument('--col_alpha', type=float, required=False, default=0.4)
    parser.add_argument('--col_dist_mul', type=float, required=False, default=1.1)
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

    state = set_state_from_config_dict({StateType.AGENT: agent_configs,
                                        StateType.OBJECT: object_configs,
                                        StateType.SIMULATOR: [simulator_config]
                                        })
    
    col_params = {'eps': args.col_eps, 'alpha': args.col_alpha, 'dist_mul': args.col_dist_mul}
    simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid, col_params)
    print('Simulator server started')
    logging.basicConfig()
    serve(simulator)