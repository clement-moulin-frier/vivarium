from contextlib import contextmanager

import numpy as np
import panel as pn

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, PointDrawTool, HoverTool, Range1d, CDSView, BooleanFilter
from param import Parameterized

from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.controllers.panel_controller import PanelController
from vivarium.simulator.states import EntityType


pn.extension()

def normal(array):
    normals = np.zeros((array.shape[0], 2))
    normals[:, 0] = np.cos(array)
    normals[:, 1] = np.sin(array)
    return normals


class EntityManager:
    def __init__(self, config, panel_configs, panel_simulator_config, selected, etype, state):
        self.config = config
        self.panel_configs = panel_configs
        self.panel_simulator_config = panel_simulator_config
        self.selected = selected
        self.etype = etype
        self.cds = ColumnDataSource(data=self.get_cds_data(state))
        self.cds.on_change('data', self.drag_cb)
        self.cds_view = self.create_cds_view()
        self.panel_simulator_config.param.watch(self.hide_all_non_existing, "hide_non_existing")
        selected.param.watch(self.update_selected_plot, ['selection'],
                             onlychanged=True, precedence=0)
        for i, pc in enumerate(self.panel_configs):
            pc.param.watch(self.update_cds_view, pc.param_names(), onlychanged=True)
            self.config[i].param.watch(self.hide_non_existing, "exists", onlychanged=False)

    def drag_cb(self, attr, old, new):
        for i, c in enumerate(self.config):
            c.x_position = new['x'][i]
            c.y_position = new['y'][i]

    @contextmanager
    def no_drag_cb(self):
        self.cds.remove_on_change('data', self.drag_cb)
        yield
        self.cds.on_change('data', self.drag_cb)

    def get_cds_data(self, state):
        raise NotImplementedError()

    def update_cds(self, state):
        self.cds.data.update(self.get_cds_data(state))

    def create_cds_view(self):
        # For each attribute in the panel config, create a filter
        # that is a logical AND of the visibility and the attribute
        return {
            attr: CDSView(filter=BooleanFilter(
                [getattr(pc, attr) and pc.visible for pc in self.panel_configs]
            )) for attr in self.panel_configs[0].param_names()
        }

    def update_cds_view(self, event):
        n = event.name
        for attr in [n] if n != "visible" else self.panel_configs[0].param_names():
            f = [getattr(pc, attr) and pc.visible for pc in self.panel_configs]
            self.cds_view[attr].filter = BooleanFilter(f)

    def update_selected_plot(self, event):
        self.cds.selected.indices = event.new

    def hide_all_non_existing(self, event):
        for i, pc in enumerate(self.panel_configs):
            if not self.config[i].exists:
                pc.visible = not event.new

    def hide_non_existing(self, event):
        if not self.panel_simulator_config.hide_non_existing:
            return
        idx = self.config.index(event.obj)
        self.panel_configs[idx].visible = event.new


    def update_selected_simulator(self):
        indices = self.cds.selected.indices
        if len(indices) > 0 and indices != self.selected.selection:
            self.selected.selection = indices

    def plot(self, fig: figure):
        raise NotImplementedError()


class AgentManager(EntityManager):
    def get_cds_data(self, state):
        pos = state.position(self.etype).center
        x, y = pos[:, 0], pos[:, 1]
        thetas = state.position(self.etype).orientation
        radii = state.diameter(self.etype) / 2.
        colors = state.agent_state.color
        motors = state.agent_state.motor
        proxs = state.agent_state.prox
        max_prox = state.agent_state.proxs_dist_max
        angle_min = np.arccos(state.agent_state.proxs_cos_min)
        wheel_diameter = state.agent_state.wheel_diameter

        # line direction
        angles = np.array(thetas)
        normals = normal(angles)

        # wheels directions
        normals_rw = normal(angles+np.pi/2)
        normals_lw = normal(angles-np.pi/2)

        # proxs directions
        normals_rp = normal(angles+np.pi/4)
        normals_lp = normal(angles-np.pi/4)

        r_wheel_x, r_wheel_y, l_wheel_x, l_wheel_y = [], [], [], []
        r_prox_x, r_prox_y, l_prox_x, l_prox_y = [], [], [], []
        orientation_lines_x, orientation_lines_y = [], []

        for xx, yy, n, nrw, nlw, nrp, nlp, r in zip(x, y, normals, normals_rw, normals_lw,
                                                    normals_rp, normals_lp, radii):
            r_wheel_x.append(xx + r * nrw[0])
            r_wheel_y.append(yy + r * nrw[1])
            l_wheel_x.append(xx + r * nlw[0])
            l_wheel_y.append(yy + r * nlw[1])

            r_prox_x.append(xx + r * nrp[0])
            r_prox_y.append(yy + r * nrp[1])
            l_prox_x.append(xx + r * nlp[0])
            l_prox_y.append(yy + r * nlp[1])

            orientation_lines_x.append([xx, xx + r * n[0]])
            orientation_lines_y.append([yy, yy + r * n[1]])

        max_angle_r = thetas - angle_min
        max_angle_l = thetas + angle_min

        data = dict(x=x, y=y, ox=orientation_lines_x, oy=orientation_lines_y, r=radii, fc=colors,
                    angle=thetas, pr=0.2*radii, rwx=r_wheel_x, rwy=r_wheel_y, lwx=l_wheel_x,
                    lwy=l_wheel_y, rwi=motors[:, 0], lwi=motors[:, 1], rpx=r_prox_x,
                    rpy=r_prox_y, lpx=l_prox_x, lpy=l_prox_y, rpi=proxs[:, 0],
                    lpi=proxs[:, 1], mar=max_angle_r, mal=max_angle_l, mpr=max_prox,
                    wd=wheel_diameter)

        return data


    def plot(self, fig: figure):
        src = {"source":self.cds}
        # wheels
        fig.rect('rwx', 'rwy', width='wd', height=1, angle='angle', fill_color='black',
                 fill_alpha='rwi', line_color=None, view=self.cds_view["visible_wheels"], **src)
        fig.rect('lwx', 'lwy', width='wd', height=1, angle='angle', fill_color='black',
                 fill_alpha='lwi', line_color=None, view=self.cds_view["visible_wheels"], **src)
        # proxs
        fig.circle('rpx', 'rpy', radius='pr', fill_color='red', fill_alpha='rpi', line_color=None,
                   view=self.cds_view["visible_proxs"], **src)
        fig.circle('lpx', 'lpy', radius='pr', fill_color='red', fill_alpha='lpi', line_color=None,
                   view=self.cds_view["visible_proxs"], **src)
        fig.wedge('x', 'y', radius='mpr', start_angle='angle', end_angle='mar', color="firebrick",
                  alpha=0.1, direction="clock", view=self.cds_view["visible_proxs"], **src)
        fig.wedge('x', 'y', radius='mpr', start_angle='angle', end_angle='mal', color="firebrick",
                  alpha=0.1, direction="anticlock", view=self.cds_view["visible_proxs"], **src)
        # direction lines
        fig.multi_line('ox', 'oy', color='black', view=self.cds_view["visible"], **src)
        # agents
        return fig.circle('x', 'y', radius='r', fill_color='fc', fill_alpha=0.6, line_color=None,
                          hover_fill_color="black", hover_fill_alpha=0.7, hover_line_color=None,
                          view=self.cds_view["visible"], **src)


class ObjectManager(EntityManager):
    def get_cds_data(self, state):
        pos = state.position(self.etype).center
        x, y = pos[:, 0], pos[:, 1]
        thetas = state.position(self.etype).orientation
        d = state.diameter(self.etype)
        colors = state.object_state.color

        data = dict(x=x, y=y, width=d, height=d, angle=thetas, fill_color=colors)
        return data

    def plot(self, fig: figure):
        src = {"source": self.cds}
        return fig.rect(x='x', y='y', width='width', height='height', angle='angle',
                        fill_color='fill_color', fill_alpha=0.6, line_color=None,
                        hover_fill_color="black", hover_fill_alpha=0.7, hover_line_color=None,
                        view=self.cds_view["visible"], **src)


class WindowManager(Parameterized):
    controller = PanelController(client=SimulatorGRPCClient())
    config_types = [key.name for key in controller.configs.keys()]
    start_toggle = pn.widgets.Toggle(**({"name": "Stop", "value": True} if controller.is_started()
                                        else {"name": "Start", "value": False}),align="center")
    entity_toggle = pn.widgets.ToggleGroup(name="EntityToggle", options=config_types,
                                           align="center", value=config_types[1:])
    update_switch = pn.widgets.Switch(name="Update plot", value=True, align="center")
    update_timestep = pn.widgets.IntSlider(name="Timestep (ms)", value=0, start=0, end=1000)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.entity_manager_classes = {EntityType.AGENT: AgentManager,
                                       EntityType.OBJECT: ObjectManager}
        self.entity_managers = {
            etype: manager_class(
                config=self.controller.configs[etype.to_state_type()],
                panel_configs=self.controller.panel_configs[etype.to_state_type()],
                panel_simulator_config=self.controller.panel_simulator_config,
                selected=self.controller.selected_entities[etype], etype=etype,
                state=self.controller.state)
                for etype, manager_class in self.entity_manager_classes.items()
        }

        self.plot = self.create_plot()
        self.app = self.create_app()
        self.set_callbacks()

    def start_toggle_cb(self, event):
        if event.new != self.controller.is_started():
            if event.new:
                self.controller.start()
            else:
                self.controller.stop()
        self.start_toggle.name = "Stop" if self.controller.is_started() else "Start"


    def entity_toggle_cb(self, event):
        for i, t in enumerate(self.config_types):
            self.config_columns[i].visible = t in event.new

    def update_timestep_cb(self, event):
        self.pcb_plot.period = event.new

    def update_plot_cb(self):
        for em in self.entity_managers.values():
            em.update_selected_simulator()
        state = self.controller.update_state()
        self.controller.pull_configs()
        if self.controller.panel_simulator_config.config_update:
            self.controller.pull_selected_configs()
        for em in self.entity_managers.values():
            with em.no_drag_cb():
                em.update_cds(state)

    def update_switch_cb(self, event):
        if event.new and not self.pcb_plot.running:
            self.pcb_plot.start()
        elif not event.new and self.pcb_plot.running:
            self.pcb_plot.stop()

    def create_plot(self):
        p_tools = "crosshair,pan,wheel_zoom,box_zoom,reset,tap,box_select,lasso_select"
        p = figure(tools=p_tools, active_drag="box_select")
        p.axis.major_label_text_font_size = "24px"
        hover = HoverTool(tooltips=None)
        p.add_tools(hover)
        p.x_range = Range1d(0, self.controller.simulator_config.box_size)
        p.y_range = Range1d(0, self.controller.simulator_config.box_size)
        draw_tool = PointDrawTool(renderers=[self.entity_managers[etype].plot(p)
                                             for etype in EntityType], add=False)
        p.add_tools(draw_tool)
        return p

    def create_app(self):
        self.config_columns = pn.Row(*
            [pn.Column(
                pn.pane.Markdown("### SIMULATOR", align="center"),
                pn.panel(self.controller.panel_simulator_config, name="Visualization configurations"),
                pn.panel(self.controller.simulator_config, name="Configurations"),
                visible=False, sizing_mode="scale_height", scroll=True)] +
            [pn.Column(
                pn.pane.Markdown(f"### {etype.name}", align="center"),
                self.controller.selected_entities[etype],
                pn.panel(self.controller.selected_panel_configs[etype], name="Visualization configurations"),
                pn.panel(self.controller.selected_configs[etype], name="State configurations"),
                visible=True, sizing_mode="scale_height", scroll=True)
            for etype in EntityType])

        app = pn.Row(pn.Column(pn.Row(pn.pane.Markdown("### Start/Stop server", align="center"),
                                      self.start_toggle),
                               pn.Row(pn.pane.Markdown("### Start/Stop update", align="center"),
                                      self.update_switch, self.update_timestep),
                               pn.panel(self.plot)),
                     pn.Column(pn.Row("### Show Configurations", self.entity_toggle),
                               pn.Row(*self.config_columns)))
        return app

    def set_callbacks(self):
        # putting directly the slider value causes bugs on some OS
        self.pcb_plot = pn.state.add_periodic_callback(self.update_plot_cb,
                                                       self.update_timestep.value or 0)
        self.entity_toggle.param.watch(self.entity_toggle_cb, "value")
        self.start_toggle.param.watch(self.start_toggle_cb, "value")
        self.update_switch.param.watch(self.update_switch_cb, "value")
        self.update_timestep.param.watch(self.update_timestep_cb, "value")
