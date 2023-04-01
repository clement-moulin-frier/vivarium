from flask import Flask, Response
import threading

from vivarium.simulator.api import serialize_state, sim_state_to_populations

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


    # Execute anything
class FlaskAppWrapper(object):
    app = None

    def __init__(self, name, simulator):
        self.app = Flask(name)
        self.simulator = simulator
        self.add_endpoint(endpoint='/is_started', endpoint_name='is_started', handler=is_started)
        self.add_endpoint(endpoint='/get_sim_config', endpoint_name='get_sim_config', handler=get_sim_config)
        self.add_endpoint(endpoint='/get_state', endpoint_name='get_state', handler=get_state)
        self.add_endpoint(endpoint='/start', endpoint_name='start', handler=start)
        self.add_endpoint(endpoint='/stop', endpoint_name='stop', handler=stop)

        threading.Thread(target=self.app.run).start()

    def run(self):
        self.app.run()

    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None):
        self.app.add_url_rule(endpoint, endpoint_name, EndpointAction(handler, self.simulator))




# a = FlaskAppWrapper('wrap')
# a.add_endpoint(endpoint='/ad', endpoint_name='ad', handler=action)
# a.run()