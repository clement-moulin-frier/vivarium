import numpy as np

# from vivarium.simulator.rest_api import SimulatorRestClient
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.simulator.config import SimulatorConfig, AgentConfig
from vivarium.simulator.simulator_controller import SimulatorController

import panel as pn
import json
import param
import time

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Button, PointDrawTool, HoverTool, Range1d
from bokeh.layouts import layout
from bokeh.events import ButtonClick


def normal(array):
    normals = np.zeros((array.shape[0], 2))
    # print('normal', array)
    normals[:, 0] = np.cos(array)
    normals[:, 1] = np.sin(array)
    return normals

pn.extension()

#simulator = SimulatorRestClient()
simulator = SimulatorController(client=SimulatorGRPCClient())

# sim_config = simulator.get_sim_config()
# print('sim_config', sim_config.param.values())
# agent_config = simulator.get_agent_config()
# population_config = simulator.get_population_config()

state = simulator.get_state()


# def push_config(e):
#     if e.name == 'entity_behaviors' and np.array_equal(e.old, e.new):
#         return
#     print('push_config', simulator.simulation_config)
#     pcb_config.stop()
#     #print(f"(event: {e.name} changed from {e.old} with type {type(e.old)} to {e.new} with type {type(e.new)}). Equals = {np.array_equal(e.old, e.new)}")
#     simulator.set_simulation_config(simulator.simulation_config)
#     # time.sleep(10)
#     pcb_config.start()
#
#
# sim_config.param.watch(push_config, sim_config.export_fields, onlychanged=True)

max_agents = 1000
all_colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(np.random.rand(max_agents) * 200 + 50, np.random.rand(max_agents) * 200 + 50)]

def get_cds_data(state):
    x = state.position[:, 0]
    y = state.position[:, 1]
    thetas = state.theta

    radius = simulator.agent_config.base_length / 2.
    colors = all_colors[:simulator.simulation_config.n_agents]  # ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)]

    normals = normal(thetas)

    orientation_lines_x = [[xx, xx + radius * n[0]] for xx, n in zip(x, normals)]
    orientation_lines_y = [[yy, yy + radius * n[1]] for yy, n in zip(y, normals)]

    return dict(x=x, y=y, ox=orientation_lines_x, oy=orientation_lines_y, r=np.ones(simulator.simulation_config.n_agents) * radius, fc=colors)


cds = ColumnDataSource(data=get_cds_data(state))


def update_cds(state):
    cds.data.update(get_cds_data(state))


TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,tap,box_select,lasso_select"

p = figure(tools=TOOLS)
p.axis.major_label_text_font_size = "24px"
hover = HoverTool(tooltips=None, mode="vline")
p.add_tools(hover)
p.x_range = Range1d(0, simulator.simulation_config.box_size)
p.y_range = Range1d(0, simulator.simulation_config.box_size)

orientations = p.multi_line('ox', 'oy', source=cds, color='black', line_width=1)
r = p.circle('x', 'y', radius='r',
             fill_color='fc', fill_alpha=0.6, line_color=None,
             hover_fill_color="black", hover_fill_alpha=0.7, hover_line_color=None, source=cds)


button = Button(name="Start" if simulator.is_started() else "Stop")


def callback(event):
    if simulator.is_started():
        button.name = "Stop"
        simulator.stop()
    else:
        button.name = "Start"
        simulator.start()


button.on_event(ButtonClick, callback)

draw_tool = PointDrawTool(renderers=[r])
p.add_tools(draw_tool)
p.toolbar.active_tap = draw_tool


row = pn.Row(p, button, simulator.simulation_config)
row.servable()


def update_plot():
    # print('update plot')
    state = simulator.get_state()
    update_cds(state)


# def pull_config():
#     changes_dict = simulator.get_recorded_changes()
#     # print('pull_config', changes_dict)
#     simulator.simulation_config.param.update(**changes_dict)


pcb_plot = pn.state.add_periodic_callback(update_plot, 10)

# pcb_config = pn.state.add_periodic_callback(pull_config, 200)


