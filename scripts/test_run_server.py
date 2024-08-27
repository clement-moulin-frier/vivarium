import logging

from vivarium.simulator.new_simulator import Simulator
from vivarium.environments.braitenberg.selective_sensing import SelectiveSensorsEnv, init_state
from vivarium.simulator.grpc_server.simulator_server import serve

lg = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

update_freq = 300

def main():
    env_state = init_state()
    env = SelectiveSensorsEnv(state=env_state)
    
    simulator = Simulator(env=env, env_state=env_state, update_freq=update_freq)
    lg.info(f"{simulator.freq = }")

    # To check if communication works, set an agent's behaviors to manual (5) and set it's motor values
    serve(simulator)
    lg.info('Simulator server started')

if __name__ == '__main__':
    main()
