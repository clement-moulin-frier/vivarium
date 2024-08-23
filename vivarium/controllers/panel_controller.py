from vivarium.controllers import converters
from vivarium.controllers.config import AgentConfig, ObjectConfig, config_to_stype, Config
from vivarium.controllers.simulator_controller import SimulatorController
# from vivarium.simulator.states import EntityType, StateType
from vivarium.simulator.new_states import EntityType, StateType
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient

import param
import numpy as np
from contextlib import contextmanager

import logging

lg = logging.getLogger(__name__)
print(f"logging in panel controller {lg = }")

class PanelConfig(Config):
    pass


class PanelEntityConfig(PanelConfig):
    visible = param.Boolean(True)


class PanelAgentConfig(PanelEntityConfig):
    visible_wheels = param.Boolean(False)
    visible_proxs = param.Boolean(False)


class PanelObjectConfig(PanelEntityConfig):
    pass


class PanelSimulatorConfig(Config):
    hide_non_existing = param.Boolean(True)
    config_update = param.Boolean(False)


panel_config_to_stype = {PanelSimulatorConfig: StateType.SIMULATOR, PanelAgentConfig: StateType.AGENT,
                         PanelObjectConfig: StateType.OBJECT}
stype_to_panel_config = {stype: config_class for config_class, stype in panel_config_to_stype.items()}


class Selected(param.Parameterized):
    selection = param.ListSelector([0], objects=[0])

    def selection_nve_idx(self, ent_idx):
        return ent_idx[np.array(self.selection)].tolist()

    def __len__(self):
        return len(self.selection)

class PanelController(SimulatorController):

    def __init__(self, **params):
        self._selected_configs_watchers = None
        self._selected_panel_configs_watchers = None
        self.selected_entities = {EntityType.AGENT: Selected(), EntityType.OBJECT: Selected()}
        self.selected_configs = {EntityType.AGENT: AgentConfig(), EntityType.OBJECT: ObjectConfig()}
        super().__init__(**params)
        self.panel_configs = {stype: [stype_to_panel_config[stype]() for _ in range(len(configs))]
                              for stype, configs in self.configs.items()}
        self.selected_panel_configs = {EntityType.AGENT: PanelAgentConfig(), EntityType.OBJECT: PanelObjectConfig()}
        self.panel_simulator_config = PanelSimulatorConfig()
        self.pull_selected_panel_configs()

        self.update_entity_list()
        for selected in self.selected_entities.values():
            selected.param.watch(self.pull_selected_configs, ['selection'], onlychanged=True, precedence=1)
            selected.param.watch(self.pull_selected_panel_configs, ['selection'], onlychanged=True)

    def watch_selected_configs(self):
        watchers = {etype: config.param.watch(self.push_selected_to_config_list, config.param_names(), onlychanged=True)
                    for etype, config in self.selected_configs.items()}
        return watchers

    def watch_selected_panel_configs(self):
        watchers = {etype: config.param.watch(self.push_selected_to_config_list, config.param_names(), onlychanged=True)
                    for etype, config in self.selected_panel_configs.items()}
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

    @contextmanager
    def dont_push_selected_panel_configs(self):
        if self._selected_panel_configs_watchers is not None:
            for etype, config in self.selected_panel_configs.items():
                config.param.unwatch(self._selected_panel_configs_watchers[etype])
        try:
            yield
        finally:
            self._selected_panel_configs_watchers = self.watch_selected_panel_configs()

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
                config_dict[etype.to_state_type()][0].idx = int(state.ent_idx(etype.to_state_type(), selected.selection[0]))
            converters.set_configs_from_state(state, config_dict)
        return state

    def pull_selected_panel_configs(self, *events):
        with self.dont_push_selected_panel_configs():
            for etype, panel_config in self.selected_panel_configs.items():
                panel_config.param.update(**self.panel_configs[etype.to_state_type()][self.selected_entities[etype].selection[0]].to_dict())

    def pull_all_data(self):
        self.pull_selected_configs()
        self.pull_configs({StateType.SIMULATOR: self.configs[StateType.SIMULATOR]})

    def push_selected_to_config_list(self, *events):
        print("push_selected_to_config_list")
        lg.info('push_selected_to_config_list %d', len(events))
        for e in events:
            if isinstance(e.obj, PanelConfig):
                stype = panel_config_to_stype[type(e.obj)]
            else:
                stype = config_to_stype[type(e.obj)]
            selected_entities = self.selected_entities[stype.to_entity_type()].selection
            for idx in selected_entities:
                if isinstance(e.obj, PanelConfig):
                    setattr(self.panel_configs[stype][idx], e.name, e.new)
                else:
                    setattr(self.configs[stype][idx], e.name, e.new)


if __name__ == '__main__':
    simulator = PanelController(client=SimulatorGRPCClient())
    simulator.configs[StateType.AGENT][0] = 'black'
    # TODO : Add a thing to check if there's a callback and see if the 

    lg.info('Done')
    