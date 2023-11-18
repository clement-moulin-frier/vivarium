from flask import Flask, Response, request
import threading
import requests
import os
from urllib.parse import urljoin
import numpy as np
import json

def serialize_state(s):
    serial_pop_kwargs = {}
    for field, jarray in s._asdict().items():
        serial_pop_kwargs[field] = np.array(jarray).tolist()

    return serial_pop_kwargs


def is_started(simulator):
    print('is_started')
    return {'is_started': simulator.is_started}

def get_sim_config(simulator):
    return simulator.simulation_config.json()

def get_agent_config(simulator):
    return simulator.agent_config.json()

def get_population_config(simulator):
    return simulator.population_config.json()

def set_population_config(simulator, json_str):
    # print("method", request.method)
    kwargs = json.loads(json_str)
    print("set_pop", json_str)
    #pop = PopulationConfig(**PopulationConfig.param.deserialize_parameters(json_str))

    #vals = dict(pop.param.values())
    #del vals['name']
    simulator.population_config.param.update(**kwargs)

    #simulator.population_config.param.update(**kwargs)


def get_state(simulator):
    return serialize_state(simulator.state)

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

    def get_agent_config(self):
        agent_config = requests.get(urljoin(self.server_url, os.path.join(self.prefix, 'get_agent_config')))
        return agent_config.json()

    def get_population_config(self):
        population_config = requests.get(urljoin(self.server_url, os.path.join(self.prefix, 'get_population_config')))
        return population_config.text  #.json()

    def set_population_config(self, **kwargs):
        #print("set_pop: before req", kwargs)
        s = json.dumps((kwargs))
        requests.get(urljoin(self.server_url, os.path.join(self.prefix, 'set_population_config')), params=dict(json_str=s))


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

    def __call__(self, **kwargs):
        print('EndpointAction', request.args)  # .to_dict())
        res = self.action(self.simulator, **request.args.to_dict())
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
        self.add_endpoint(endpoint='/get_agent_config', endpoint_name='get_agent_config', handler=get_agent_config)
        self.add_endpoint(endpoint='/get_population_config', endpoint_name='get_population_config', handler=get_population_config)
        self.add_endpoint(endpoint='/set_population_config', endpoint_name='set_population_config', handler=set_population_config)
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