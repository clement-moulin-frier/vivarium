from flask import Flask, Response
import threading
import requests
import os
from urllib.parse import urljoin

from vivarium.simulator.api import serialize_state, sim_state_to_populations



# There could be a way to have i/o type conversions to rest in __call__ above (or in the class). This way, all functions
# below can be used in a client cla
# But how about requests for setters (e.g. set_state)
# Might be a weird solution ...

def is_started(simulator):
    print('is_started')
    return {'is_started': simulator.is_started}

def get_sim_config(simulator):
    return simulator.sim_config.get_sim_config()

def get_state(simulator):
    return serialize_state(sim_state_to_populations(simulator.state, simulator.entity_slices))

def start(simulator):
    simulator.run(threaded=True)

def stop(simulator):
    simulator.stop()


class SimulatorRestClient:
    def __init__(self, sim_server_url='http://127.0.0.1:5000'):
        self.server_url = sim_server_url
        self.prefix = ''
    def get_sim_config(self):
        sim_config = requests.get(urljoin(self.server_url, os.path.join(self.prefix, 'get_sim_config')))
        return sim_config.json()

    def get_state(self):
        state = requests.post(urljoin(self.server_url, os.path.join(self.prefix, 'get_state')))
        return state.json()
    # def run(self):
    #     requests.get(urljoin(self.server_url, 'run'))
    def start(self):
        requests.get(urljoin(self.server_url, os.path.join(self.prefix, 'start')))

    def stop(self):
        requests.get(urljoin(self.server_url, os.path.join(self.prefix, 'stop')))

    def is_started(self):
        #print(urljoin(self.server_url, os.path.join(self.prefix, 'is_started')))
        req = requests.get(urljoin(self.server_url, os.path.join(self.prefix, 'is_started')))
        return req.json()['is_started']
#
#     def set_motors(self, agent_idx, motors):
#         args = {'agent_idx': agent_idx, 'motors': motors}
#         requests.post(urljoin(self.server_url, 'set_motors'), data=args)
#
#     def no_set_motors(self):
#         requests.get(urljoin(self.server_url, 'no_set_motors'))
#     def get_motors(self):
#         req = requests.post(urljoin(self.server_url, 'get_motors'))
#         return req.json()

class EndpointAction(object):

    def __init__(self, action, simulator):
        self.action = action
        self.simulator = simulator
        self.response = Response(status=200, headers={})

    def __call__(self, *args):
        res = self.action(self.simulator)
        if res is None:
            return self.response
        else:
            return res

class FlaskAppWrapper(object):
    app = None

    def __init__(self, name, simulator):
        self.app = Flask(name)
        self.simulator = simulator
        self.add_endpoint(endpoint='/is_started', endpoint_name='is_started', handler=is_started)
        self.add_endpoint(endpoint='/get_sim_config', endpoint_name='get_sim_config', handler=get_sim_config)
        self.add_endpoint(endpoint='/get_state', endpoint_name='get_state', handler=get_state, methods=['POST'])
        self.add_endpoint(endpoint='/start', endpoint_name='start', handler=start)
        self.add_endpoint(endpoint='/stop', endpoint_name='stop', handler=stop)

        threading.Thread(target=self.app.run).start()

    def run(self):
        self.app.run()

    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None, methods=['GET']):
        self.app.add_url_rule(endpoint, endpoint_name, EndpointAction(handler, self.simulator), methods=methods)




# a = FlaskAppWrapper('wrap')
# a.add_endpoint(endpoint='/ad', endpoint_name='ad', handler=action)
# a.run()