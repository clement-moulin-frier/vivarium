from vivarium.controllers import converters
from vivarium.controllers.config import AgentConfig, ObjectConfig, config_to_etype
from vivarium.controllers.simulator_controller import SimulatorController
from vivarium.simulator.sim_computation import EntityType
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient

import param
import numpy as np
from contextlib import contextmanager


class Selected(param.Parameterized):
    selection = param.ListSelector([0], objects=[0])

    def selection_nve_idx(self, nve_idx):
        return nve_idx[np.array(self.selection)].tolist()


class PanelController(SimulatorController):
    selected_entities = param.Dict({EntityType.AGENT: Selected(), EntityType.OBJECT: Selected()})
    selected_configs = param.Dict({EntityType.AGENT: AgentConfig(), EntityType.OBJECT: ObjectConfig()})

    def __init__(self, **params):
        self._selected_configs_watchers = None
        super().__init__(**params)
        self.update_entity_list()
        for etype, selected in self.selected_entities.items():
            selected.param.watch(self.pull_selected_configs, ['selection'], onlychanged=True, precedence=1)
        self.simulation_config.param.watch(self.update_entity_list, ['n_agents', 'n_objects'], onlychanged=True)

    def watch_selected_configs(self):
        watchers = {etype: config.param.watch(self.push_selected_to_state, config.param_names(), onlychanged=True)
                    for etype, config in self.selected_configs.items()}
        return watchers

    @contextmanager
    def dont_push_selected_configs(self):
        if self._selected_configs_watchers is not None:
            for etype, config in self.selected_configs.items():
                config.param.unwatch(self._selected_configs_watchers[etype])
        try:
            yield
        finally:
            self._selected_configs_watchers = self.watch_selected_configs()

    def update_entity_list(self, *events):
        state = self.state
        for etype, selected in self.selected_entities.items():
            selected.param.selection.objects = state.entity_idx(etype).tolist()

    def pull_selected_configs(self, *events):
        state = self.state
        config_dict = {etype: [config] for etype, config in self.selected_configs.items()}
        with self.dont_push_selected_configs():
            for etype, selected in self.selected_entities.items():
                config_dict[etype][0].idx = state.nve_idx(etype, selected.selection[0])
            converters.set_configs_from_state(state, config_dict)
        return state

    def pull_all_data(self):
        self.pull_selected_configs()
        self.pull_simulation_config()

    def push_selected_to_state(self, *events):
        print('push_selected_to_state', len(events))
        for e in events:
            etype = config_to_etype[type(e.obj)]
            selected_entities = self.selected_entities[etype].selection
            for idx in selected_entities:
                setattr(self.entity_configs[etype][idx], e.name, e.new)


if __name__ == '__main__':
    simulator = PanelController(client=SimulatorGRPCClient())
    print('Done')
    