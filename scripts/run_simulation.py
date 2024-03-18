import argparse
import logging

import numpy as np
import jax.numpy as jnp 

from vivarium.simulator import behaviors
from vivarium.simulator.sim_computation import dynamics_rigid
from vivarium.simulator.states import SimulatorState, AgentState, ObjectState, NVEState, State
from vivarium.simulator.states import init_simulator_state, init_agent_state, init_object_state, init_nve_state, init_state

from vivarium.controllers.config import AgentConfig, ObjectConfig, SimulatorConfig
from vivarium.controllers import converters
from vivarium.simulator.simulator import Simulator

lg = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Simulator Configuration')
    # Experiment run arguments
    parser.add_argument('--num_steps', type=int, default=10, help='Number of simulation loops')
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
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=args.log_level.upper())
    
    # TODO : set the state without the configs 

    simulator_state = init_simulator_state(
        box_size=args.box_size,
        n_agents=args.n_agents,
        n_objects=args.n_objects,
        num_steps_lax=args.num_steps_lax,
        neighbor_radius=args.neighbor_radius,
        dt=args.dt,
        to_jit=args.to_jit,
        use_fori_loop=args.use_fori_loop
    )

    agents_state = init_agent_state(
        n_agents=args.n_agents,
    )

    object_state = init_object_state(
        n_objects=args.n_objects,
    )

    nve_state = init_nve_state(
        simulator_state=simulator_state,
        diameter=diameter,
        friction=friction,
        seed=0
    )


    simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid)

    lg.info("Running simulation")

    simulator.run(threaded=False, num_steps=args.num_steps)

    lg.info("Simulation complete")
