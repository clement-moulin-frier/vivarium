import argparse
import logging

from vivarium.simulator import behaviors
from vivarium.simulator.states import init_simulator_state
from vivarium.simulator.states import init_agent_state
from vivarium.simulator.states import init_object_state
from vivarium.simulator.states import init_nve_state
from vivarium.simulator.states import init_state
from vivarium.simulator.simulator import Simulator
from vivarium.simulator.sim_computation import dynamics_rigid

lg = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Simulator Configuration')
    # Experiment run arguments
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of simulation steps')
    # Simulator config arguments
    parser.add_argument('--box_size', type=float, default=100.0, help='Size of the simulation box')
    parser.add_argument('--max_agents', type=int, default=10, help='Number of agents')
    parser.add_argument('--max_objects', type=int, default=2, help='Number of objects')
    parser.add_argument('--num_steps_lax', type=int, default=4, help='Number of lax steps per loop')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step size')
    parser.add_argument('--freq', type=float, default=40.0, help='Frequency parameter')
    parser.add_argument('--neighbor_radius', type=float, default=100.0, help='Radius for neighbor calculations')
    # By default jit compile the code and use normal python loops
    parser.add_argument('--to_jit', action='store_false', help='Whether to use JIT compilation')
    parser.add_argument('--use_fori_loop', action='store_true', help='Whether to use fori loop')
    parser.add_argument('--collision_eps', type=float, required=False, default=0.1)
    parser.add_argument('--collision_alpha', type=float, required=False, default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=args.log_level.upper())
    
    simulator_state = init_simulator_state(
        box_size=args.box_size,
        max_agents=args.max_agents,
        max_objects=args.max_objects,
        num_steps_lax=args.num_steps_lax,
        neighbor_radius=args.neighbor_radius,
        dt=args.dt,
        freq=args.freq,
        to_jit=args.to_jit,
        use_fori_loop=args.use_fori_loop,
        collision_eps=args.collision_eps,
        collision_alpha=args.collision_alpha
    )

    agents_state = init_agent_state(simulator_state=simulator_state)

    objects_state = init_object_state(simulator_state=simulator_state)

    nve_state = init_nve_state(simulator_state=simulator_state)

    state = init_state(
        simulator_state=simulator_state,
        agents_state=agents_state,
        objects_state=objects_state,
        nve_state=nve_state
        )

    simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid)

    lg.info("Running simulation")

    simulator.run(threaded=False, num_steps=args.num_steps)

    lg.info("Simulation complete")
