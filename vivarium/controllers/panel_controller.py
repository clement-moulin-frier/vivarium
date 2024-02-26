from vivarium.controllers import converters
from vivarium.controllers.config import AgentConfig, ObjectConfig, config_to_stype
from vivarium.controllers.simulator_controller import SimulatorController
from vivarium.simulator.sim_computation import EntityType, StateType
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient

import param
import numpy as np
from contextlib import contextmanager

import logging

logging.basicConfig(level=logging.INFO)
lg = logging.getLogger(__name__)

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
        config_dict = {etype.to_state_type(): [config] for etype, config in self.selected_configs.items()}
        with self.dont_push_selected_configs():
            # Todo: check if for loop below is still required
            for etype, selected in self.selected_entities.items():
                config_dict[etype.to_state_type()][0].idx = int(state.nve_idx(etype.to_state_type(), selected.selection[0]))
            converters.set_configs_from_state(state, config_dict)
        return state

    def pull_all_data(self):
        self.pull_selected_configs()
        self.pull_configs({StateType.SIMULATOR: self.configs[StateType.SIMULATOR]})

    def push_selected_to_state(self, *events):
        lg.info('push_selected_to_state %d', len(events))
        for e in events:
            stype = config_to_stype[type(e.obj)]
            selected_entities = self.selected_entities[stype.to_entity_type()].selection
            for idx in selected_entities:
                setattr(self.configs[stype][idx], e.name, e.new)


if __name__ == '__main__':
    simulator = PanelController(client=SimulatorGRPCClient())
    lg.info('Done')
    