import numpy as np
import panel as pn

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, PointDrawTool, HoverTool, Range1d, CDSView, BooleanFilter

from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.controllers.panel_controller import PanelController
from vivarium.simulator.sim_computation import EntityType


pn.extension()


def normal(array):
    normals = np.zeros((array.shape[0], 2))
    normals[:, 0] = np.cos(array)
    normals[:, 1] = np.sin(array)
    return normals


class EntityManager:
    def __init__(self, config, panel_configs, selected, etype, state):
        self.config = config
        self.panel_configs = panel_configs
        self.selected = selected
        self.etype = etype
        self.cds = ColumnDataSource(data=self.get_cds_data(state))
        self.cds_view = self.create_cds_view()
        selected.param.watch(self.update_selected_plot, ['selection'],
                             onlychanged=True, precedence=0)
        for pc in self.panel_configs:
            pc.param.watch(self.update_cds_view, pc.param_names())

    def get_cds_data(self, state):
        raise NotImplementedError()

    def update_cds(self, state):
        self.cds.data.update(self.get_cds_data(state))

    def create_cds_view(self):
        visible = {"all":[], "proxs":[], "wheels":[]}
        for c in self.panel_configs:
            if "visible" in c.param:
                visible["all"].append(c.visible)
            if "visible_proxs" in c.param:
                visible["proxs"].append(c.visible_proxs and c.visible)
            if "visible_wheels" in c.param:
                visible["wheels"].append(c.visible_wheels and c.visible)
        return {k:CDSView(filter=BooleanFilter(v)) for k, v in visible.items()}

    def update_cds_view(self, event):
        n = event.name
        if n == "visible":
            f = [c.visible for c in self.panel_configs]
            self.cds_view["all"].filter = BooleanFilter(f)
        elif n == "visible_wheels":
            f = [c.visible_wheels for c in self.panel_configs]
            self.cds_view["wheels"].filter = BooleanFilter(f)
        elif n == "visible_proxs":
            f = [c.visible_proxs for c in self.panel_configs]
            self.cds_view["proxs"].filter = BooleanFilter(f)

    def update_selected_plot(self, event):
        self.cds.selected.indices = event.new

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
                    lpi=proxs[:, 1], mar=max_angle_r, mal=max_angle_l, mpr=max_prox)

        return data


    def plot(self, fig: figure):
        src = {"source":self.cds}
        # wheels
        fig.rect('rwx', 'rwy', width=3, height=1, angle='angle', fill_color='black',
                 fill_alpha='rwi', line_color=None, view=self.cds_view["wheels"], **src)
        fig.rect('lwx', 'lwy', width=3, height=1, angle='angle', fill_color='black',
                 fill_alpha='lwi', line_color=None, view=self.cds_view["wheels"], **src)
        # proxs
        fig.circle('rpx', 'rpy', radius='pr', fill_color='red', fill_alpha='rpi', line_color=None,
                   view=self.cds_view["proxs"], **src)
        fig.circle('lpx', 'lpy', radius='pr', fill_color='red', fill_alpha='lpi', line_color=None,
                   view=self.cds_view["proxs"], **src)
        fig.wedge('x', 'y', radius='mpr', start_angle='angle', end_angle='mar', color="firebrick",
                  alpha=0.1, direction="clock", view=self.cds_view["proxs"], **src)
        fig.wedge('x', 'y', radius='mpr', start_angle='angle', end_angle='mal', color="firebrick",
                  alpha=0.1, direction="anticlock", view=self.cds_view["proxs"], **src)
        # direction lines
        fig.multi_line('ox', 'oy', color='black', line_width=1, view=self.cds_view["all"], **src)
        # agents
        return fig.circle('x', 'y', radius='r', fill_color='fc', fill_alpha=0.6, line_color=None,
                          hover_fill_color="black", hover_fill_alpha=0.7, hover_line_color=None,
                          view=self.cds_view["all"], **src)


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
                        view=self.cds_view["all"], **src)


simulator = PanelController(client=SimulatorGRPCClient())
sim_state = simulator.state

entity_types = [EntityType.AGENT, EntityType.OBJECT]

entity_manager_classes = {EntityType.AGENT: AgentManager, EntityType.OBJECT: ObjectManager}

entity_managers = {
    etype: manager_class(
        config=simulator.selected_configs[etype],
        panel_configs=simulator.panel_configs[etype.to_state_type()],
        selected=simulator.selected_entities[etype], etype=etype, state=sim_state)
        for etype, manager_class in entity_manager_classes.items()
}

TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,tap,box_select,lasso_select"

p = figure(tools=TOOLS)
p.axis.major_label_text_font_size = "24px"
hover = HoverTool(tooltips=None)
p.add_tools(hover)
p.x_range = Range1d(0, simulator.simulator_config.box_size)
p.y_range = Range1d(0, simulator.simulator_config.box_size)

start_toggle = pn.widgets.Toggle(**({"name": "Stop", "value": True} if simulator.is_started()
                                    else {"name": "Start", "value": False}), align="center")


def callback(event):
    if event.type != "changed":
        return
    if simulator.is_started():
        simulator.stop()
        start_toggle.name = "Start"
    else:
        simulator.start()
        start_toggle.name = "Stop"

start_toggle.param.watch(callback, "value")

draw_tool = PointDrawTool(renderers=[entity_managers[etype].plot(p) for etype in entity_types])
p.add_tools(draw_tool)
# p.toolbar.active_tap = draw_tool

# https://panel.holoviz.org/how_to/param/custom.html
sim_panel = pn.Param(simulator.param)

# Selector for entity attributes
config_columns = pn.Row(*[
    pn.Column(
        simulator.selected_entities[etype],
        simulator.selected_panel_configs[etype],
        simulator.selected_configs[etype],
        visible=False, sizing_mode="stretch_width", scroll=True)
    for etype in EntityType])
config_columns.append(pn.Column(simulator.simulator_config, visible=False, sizing_mode="scale_both",
                                scroll=True))

config_types = ["Agents", "Objects", "Simulator"]
entity_toggle = pn.widgets.ToggleGroup(name="EntityToggle", options=config_types, align="center")

def toggle_callback(event):
    for i, t in enumerate(config_types):
        config_columns[i].visible = t in event.new

entity_toggle.param.watch(toggle_callback, "value")

update_switch = pn.widgets.Switch(name="Update plot", value=True, align="center")
update_timestep = pn.widgets.IntSlider(name="Timestep (ms)", value=10, start=0, end=1000)


app = pn.Row(pn.Column(pn.Row(pn.pane.Markdown("### Start/Stop server", align="center"),
                              start_toggle),
                       pn.Row(pn.pane.Markdown("### Start/Stop update", align="center"),
                              update_switch, update_timestep),
                       pn.panel(p)),
             pn.Column(pn.Row("### Show Configs",entity_toggle),
                       pn.Row(*config_columns)))

app.servable()


def update_plot():
    for em in entity_managers.values():
        em.update_selected_simulator()
    state = simulator.update_state()
    simulator.pull_configs()
    for em in entity_managers.values():
        em.update_cds(state)


pcb_plot = pn.state.add_periodic_callback(update_plot, 10)

def timestep_callback(event):
    pcb_plot.period = event.new

def plot_update_callback(event):
    if event.new and not pcb_plot.running:
        pcb_plot.start()
    elif not event.new and pcb_plot.running:
        pcb_plot.stop()

update_switch.param.watch(plot_update_callback, "value")
update_timestep.param.watch(timestep_callback, "value")
