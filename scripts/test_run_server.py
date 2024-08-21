import logging

from vivarium.simulator.new_simulator import Simulator
from vivarium.environments.braitenberg.selective_sensing import SelectiveSensorsEnv, init_state
from vivarium.simulator.grpc_server.simulator_server import serve

lg = logging.getLogger(__name__)

def main():
    logging.basicConfig(level='INFO')

    env_state = init_state()
    env = SelectiveSensorsEnv(state=env_state)
    
    simulator = Simulator(env=env, env_state=env_state)
    sim_state = simulator.step()

    # print(f"{sim_state.agent_state = }")
    print(f"{sim_state.agent_state.proximity_map_dist.shape = }") # (10, 20)
    print(f"{sim_state.agent_state.sensed.shape = }") # (10, 2, 4) --> Might need to flatten it ? idk
    # seems to work at the moment
    lg.info('Simulator server started')
    serve(simulator)

if __name__ == '__main__':
    main()
