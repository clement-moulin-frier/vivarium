import numpy as np
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.controllers.panel_controller import PanelController
from vivarium.simulator.sim_computation import EntityType

import panel as pn

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, PointDrawTool, HoverTool, Range1d

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
        selected.param.watch(self.update_selected_plot, ['selection'], onlychanged=True, precedence=0)

    def get_cds_data(self, state):
        raise NotImplementedError()

    def update_cds(self, state):
        self.cds.data.update(self.get_cds_data(state))

    def update_selected_plot(self, event):
        self.cds.selected.indices = event.new

    def update_selected_simulator(self):
        if len(self.cds.selected.indices) > 0 and self.cds.selected.indices != self.selected.selection:
            self.selected.selection = self.cds.selected.indices

    def plot(self, fig: figure):
        raise NotImplementedError()


class AgentManager(EntityManager):
    def get_cds_data(self, state):
        pos = state.position(self.etype).center
        x, y = pos[:, 0], pos[:, 1]
        thetas = state.position(self.etype).orientation
        radii = state.diameter(self.etype) / 2.
        colors = state.agent_state.color  # ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)]
        motors = state.agent_state.motor
        sensors = state.agent_state.prox
        max_sensor = state.agent_state.proxs_dist_max
        angle_min = np.arccos(state.agent_state.proxs_cos_min)
        visible = np.where(state.nve_state.visible & (state.nve_state.entity_type == self.etype.value))[0]
        
        # line direction
        angles = np.array(thetas)
        normals = normal(angles)

        # wheels directions
        normals_rw = normal(angles+np.pi/2)
        normals_lw = normal(angles-np.pi/2)

        # sensors directions
        normals_rs = normal(angles+np.pi/4)
        normals_ls = normal(angles-np.pi/4)

        r_wheel_x = [xx + r * n[0] for xx, n, r in zip(x, normals_rw, radii)]
        r_wheel_y = [yy + r * n[1] for yy, n, r in zip(y, normals_rw, radii)]
        l_wheel_x = [xx + r * n[0] for xx, n, r in zip(x, normals_lw, radii)]
        l_wheel_y = [yy + r * n[1] for yy, n, r in zip(y, normals_lw, radii)]

        r_sensor_x = [xx + r * n[0] for xx, n, r in zip(x, normals_rs, radii)]
        r_sensor_y = [yy + r * n[1] for yy, n, r in zip(y, normals_rs, radii)]
        l_sensor_x = [xx + r * n[0] for xx, n, r in zip(x, normals_ls, radii)]
        l_sensor_y = [yy + r * n[1] for yy, n, r in zip(y, normals_ls, radii)]
        max_angle_r = thetas - angle_min
        max_angle_l = thetas + angle_min

        orientation_lines_x = [[xx, xx + r * n[0]] for xx, n, r in zip(x, normals, radii)]
        orientation_lines_y = [[yy, yy + r * n[1]] for yy, n, r in zip(y, normals, radii)]

        data = dict(x=x, y=y, ox=orientation_lines_x, oy=orientation_lines_y, r=radii, fc=colors, angle=thetas, sr=0.2*radii,
                    rwx=r_wheel_x, rwy=r_wheel_y, lwx=l_wheel_x, lwy=l_wheel_y, rwi=motors[:, 0], lwi=motors[:, 1],
                    rsx=r_sensor_x, rsy=r_sensor_y, lsx=l_sensor_x, lsy=l_sensor_y, rsi=sensors[:, 0], lsi=sensors[:, 1],
                    mar=max_angle_r, mal=max_angle_l, mr=max_sensor)
        
        return {k: np.array(v)[visible] for k, v in data.items()}

    def plot(self, fig: figure):
        src = {"source":self.cds}
        # wheels
        fig.rect('rwx', 'rwy', width=3, height=1, angle='angle', fill_color='black', fill_alpha='rwi', line_color=None, **src)
        fig.rect('lwx', 'lwy', width=3, height=1, angle='angle', fill_color='black', fill_alpha='lwi', line_color=None, **src)
        # sensors
        fig.circle('rsx', 'rsy', radius='sr', fill_color='red', fill_alpha='rsi', line_color=None, **src)
        fig.circle('lsx', 'lsy', radius='sr', fill_color='red', fill_alpha='lsi', line_color=None, **src)
        fig.wedge('x', 'y', radius='mr', start_angle='angle', end_angle='mar', color="firebrick", alpha=0.1, direction="clock", **src)
        fig.wedge('x', 'y', radius='mr', start_angle='angle', end_angle='mal', color="firebrick", alpha=0.1, direction="anticlock", **src)
        # direction lines
        fig.multi_line('ox', 'oy', color='black', line_width=1, **src)
        # agents
        return fig.circle('x', 'y', radius='r', fill_color='fc', fill_alpha=0.6, line_color=None,
                          hover_fill_color="black", hover_fill_alpha=0.7, hover_line_color=None, **src)
        

class ObjectManager(EntityManager):
    def get_cds_data(self, state):
        pos = state.position(self.etype).center
        x, y = pos[:, 0], pos[:, 1]
        thetas = state.position(self.etype).orientation
        d = state.diameter(self.etype)
        colors = state.object_state.color
        visible = state.nve_state.visible[state.nve_state.entity_type == self.etype.value].astype(bool)

        data = dict(x=x, y=y, width=d, height=d, angle=thetas, fill_color=colors, v=visible)
        return {k: np.array(v)[visible] for k, v in data.items()}

    def plot(self, fig: figure):
        src = {"source": self.cds}
        return fig.rect(x='x', y='y', width='width', height='height', angle='angle', fill_color='fill_color',
                        fill_alpha=0.6, line_color=None, hover_fill_color="black", hover_fill_alpha=0.7,
                        hover_line_color=None, **src)


simulator = PanelController(client=SimulatorGRPCClient())
state = simulator.state

entity_types = [EntityType.AGENT, EntityType.OBJECT]

entity_manager_classes = {EntityType.AGENT: AgentManager, EntityType.OBJECT: ObjectManager}

entity_managers = {etype: manager_class(config=simulator.selected_configs[etype],
                                        panel_configs=simulator.panel_configs[etype.to_state_type()],
                                        selected=simulator.selected_entities[etype], etype=etype, state=state)
                   for etype, manager_class in entity_manager_classes.items()}

TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,tap,box_select,lasso_select"

p = figure(tools=TOOLS)
p.axis.major_label_text_font_size = "24px"
hover = HoverTool(tooltips=None)
p.add_tools(hover)
p.x_range = Range1d(0, simulator.simulator_config.box_size)
p.y_range = Range1d(0, simulator.simulator_config.box_size)

start_toggle = pn.widgets.Toggle(**({"name": "Stop", "value": True} if simulator.is_started()
                                    else {"name": "Start", "value": False}))


def callback(event):
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
entity_columns = [pn.Column(simulator.selected_entities[etype], simulator.selected_panel_configs[etype],
                            simulator.selected_configs[etype], visible=False) for etype in EntityType]

entity_toggle = pn.widgets.ToggleGroup(name="EntityToggle", options=[t.name for t in entity_types])

def toggle_callback(event):
    for i, t in enumerate(entity_types):
        entity_columns[i].visible = (t.name in event.new)


entity_toggle.param.watch(toggle_callback, "value")

# Switch for simulator settings
settings_pane = pn.panel(simulator.simulator_config, visible=False)

settings_switch = pn.widgets.Switch(name='Simulator settings')


def settings_callback(event):
    settings_pane.visible = event.new


settings_switch.param.watch(settings_callback, "value")


update_slider = pn.widgets.IntSlider(value=10, start=0, end=1000, step=10, name="Update frequency (ms)")


app = pn.Row(pn.Column(pn.Row(start_toggle, update_slider),
                       p,
                       pn.Row("Show sim settings", settings_switch),
                       settings_pane),
             pn.Column(pn.Row("### Show attributes", entity_toggle),
                       pn.Row(*entity_columns)))

app.servable()


def update_plot():
    for em in entity_managers.values():
        em.update_selected_simulator()
    state = simulator.update_state()
    simulator.pull_configs()
    for em in entity_managers.values():
        em.update_cds(state)


pcb_plot = pn.state.add_periodic_callback(update_plot, 10)


def update_freq(event):
    pcb_plot.period = event.new
    if event.new == 0 and pcb_plot.running:
        pcb_plot.stop()
    elif not pcb_plot.running:
        pcb_plot.start()


update_slider.param.watch(update_freq, "value")