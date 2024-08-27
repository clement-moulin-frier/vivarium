import logging

from vivarium.simulator.new_simulator import Simulator
from vivarium.environments.braitenberg.selective_sensing import SelectiveSensorsEnv, init_state
from vivarium.simulator.grpc_server.simulator_server import serve

lg = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

update_freq = 1
update_freq = 300

def main():
    # Add custom entities data to debug the client server communication
    entities_sbutypes = ['PREYS', 'PREDS', 'RESSOURCES', 'POISON']

    preys_data = {
        'type': 'AGENT',
        'num': 1,
        'color': 'blue',
        'selective_behaviors': {
            'love': {'beh': 'FEAR', 'sensed': ['PREYS', 'PREDS', 'RESSOURCES', 'POISON']},
            'fear': {'beh': 'FEAR', 'sensed': []}
        }}

    preds_data = {
        'type': 'AGENT',
        'num': 3,
        'color': 'red',
        'selective_behaviors': {
            'aggr': {'beh': 'AGGRESSION','sensed': ['PREYS']},
            'fear': {'beh': 'FEAR','sensed': ['POISON']
            }
        }}

    ressources_data = {
        'type': 'OBJECT',
        'num': 2,
        'color': 'green'}

    poison_data = {
        'type': 'OBJECT',
        'num': 2,
        'color': 'purple'}

    entities_data = {
        'EntitySubTypes': entities_sbutypes,
        'Entities': {
            'PREYS': preys_data,
            'PREDS': preds_data,
            'RESSOURCES': ressources_data,
            'POISON': poison_data
        }}


    env_state = init_state(entities_data=entities_data)
    env = SelectiveSensorsEnv(state=env_state)
    
    simulator = Simulator(env=env, env_state=env_state, update_freq=update_freq)
    lg.info(f"{simulator.freq = }")
    sim_state = simulator.step()

    lg.info(f"{sim_state.agent_state.proximity_map_dist.shape = }") # (10, 20)
    lg.info(f"{sim_state.agent_state.sensed.shape = }") # (10, 2, 4) --> Might need to flatten it ? idk
    
    # To check if communication works, set an agent's behaviors to manual (5) and set it's motor values
    serve(simulator)
    lg.info('Simulator server started')

if __name__ == '__main__':
    main()
