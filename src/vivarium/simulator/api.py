import requests
from urllib.parse import urljoin


sim_server_url = 'http://127.0.0.1:5000' #'http://localhost:8086'

class Simulator:
    def __init__(self, sim_server_url=sim_server_url):
        self.server_url = sim_server_url
    def get_sim_config(self):
        sim_config = requests.get(urljoin(self.server_url, 'get_sim_config'))
        return sim_config.json()

    def get_sim_state(self):
        state = requests.post(urljoin(self.server_url, 'get_state'))
        return state.json()
    def run(self):
        requests.get(urljoin(self.server_url, 'run'))
    def start_sim(self):
        requests.get(urljoin(self.server_url, 'start'))

    def stop_sim(self):
        requests.get(urljoin(self.server_url, 'stop'))

    def is_started(self):
        req = requests.get(urljoin(self.server_url, 'is_started'))
        return req.json()['is_started']

    def set_motors(self, agent_idx, motors):
        args = {'agent_idx': agent_idx, 'motors': motors}
        requests.post(urljoin(self.server_url, 'set_motors'), data=args)

    def no_set_motors(self):
        requests.get(urljoin(self.server_url, 'no_set_motors'))
    def get_motors(self):
        req = requests.post(urljoin(self.server_url, 'get_motors'))
        return req.json()


#@tranquilize()
@app.route("/get_sim_config", methods=["GET"])
def get_sim_config():

    sim_config = dict(
        box_size=box_size,
        map_dim=map_dim,
        wheel_diameter=wheel_diameter,
        base_lenght=base_lenght,
        pop_config={e_type.name: n for e_type, n in pop_config.items()}
    )
    return sim_config
