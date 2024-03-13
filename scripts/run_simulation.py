import argparse
import logging

import numpy as np

from vivarium.simulator import behaviors
from vivarium.simulator.sim_computation import dynamics_rigid, StateType
from vivarium.controllers.config import AgentConfig, ObjectConfig, SimulatorConfig
from vivarium.controllers import converters
from vivarium.simulator.simulator import Simulator

lg = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Simulator Configuration')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')
    # Experiment run arguments
    parser.add_argument('--num_loops', type=int, default=10, help='Number of simulation loops')
    # Simulator config arguments
    parser.add_argument('--box_size', type=float, default=100.0, help='Size of the simulation box')
    parser.add_argument('--n_agents', type=int, default=10, help='Number of agents')
    parser.add_argument('--n_objects', type=int, default=2, help='Number of objects')
    parser.add_argument('--num_steps_lax', type=int, default=4, help='Number of lax steps per loop')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step size')
    parser.add_argument('--freq', type=float, default=40.0, help='Frequency parameter')
    parser.add_argument('--neighbor_radius', type=float, default=100.0, help='Radius for neighbor calculations')
    # By default jit compile the code and use normal python loops
    parser.add_argument('--to_jit', action='store_false', help='Whether to use JIT compilation')
    parser.add_argument('--use_fori_loop', action='store_true', help='Whether to use fori loop')
    parser.add_argument('--col_eps', type=float, required=False, default=0.003)
    parser.add_argument('--col_alpha', type=float, required=False, default=0.7)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=args.log_level.upper())

    simulator_config = SimulatorConfig(
        box_size=args.box_size,
        n_agents=args.n_agents,
        n_objects=args.n_objects,
        num_steps_lax=args.num_steps_lax,
        dt=args.dt,
        freq=args.freq,
        neighbor_radius=args.neighbor_radius,
        to_jit=args.to_jit,
        use_fori_loop=args.use_fori_loop,
        col_eps=args.col_eps,
        col_alpha=args.col_alpha
    )
    
    agent_configs = [
        AgentConfig(idx=i,
                    x_position=np.random.rand() * simulator_config.box_size,
                    y_position=np.random.rand() * simulator_config.box_size,
                    orientation=np.random.rand() * 2. * np.pi)
        for i in range(simulator_config.n_agents)
        ]

    object_configs = [
        ObjectConfig(idx=simulator_config.n_agents + i,
                    x_position=np.random.rand() * simulator_config.box_size,
                    y_position=np.random.rand() * simulator_config.box_size,
                    orientation=np.random.rand() * 2. * np.pi)
        for i in range(simulator_config.n_objects)
        ]

    state = converters.set_state_from_config_dict(
        {
            StateType.AGENT: agent_configs,
            StateType.OBJECT: object_configs,
            StateType.SIMULATOR: [simulator_config]
        }
        )


    simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid)

    lg.info("Running simulation")

    simulator.run(threaded=False, num_loops=10)

    lg.info("Simulation complete")