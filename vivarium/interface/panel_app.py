import numpy as np

from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.simulator.simulator_controller import SimulatorController

import panel as pn

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Button, PointDrawTool, HoverTool, Range1d
from bokeh.events import ButtonClick


def normal(array):
    normals = np.zeros((array.shape[0], 2))
    normals[:, 0] = np.cos(array)
    normals[:, 1] = np.sin(array)
    return normals


pn.extension()

simulator = SimulatorController(client=SimulatorGRPCClient())

state = simulator.get_nve_state()

max_agents = 1000
all_colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(np.random.rand(max_agents) * 200 + 50, np.random.rand(max_agents) * 200 + 50)]


def get_cds_data(state):
    pos = state.position.center
    x, y = pos[:, 0], pos[:, 1]
    thetas = state.position.orientation
    radii = state.base_length / 2.
    n_agents = x.shape[0]
    colors = all_colors[:n_agents]  # ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)]

    normals = normal(thetas)

    orientation_lines_x = [[xx, xx + r * n[0]] for xx, n, r in zip(x, normals, radii)]
    orientation_lines_y = [[yy, yy + r * n[1]] for yy, n, r in zip(y, normals, radii)]

    return dict(x=x, y=y, ox=orientation_lines_x, oy=orientation_lines_y, r=radii, fc=colors)


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


# https://panel.holoviz.org/how_to/param/custom.html
motors = pn.Param(simulator.param,
                  widgets={'left_motor': pn.widgets.FloatSlider,  # {'widget_type': pn.widgets.FloatSlider, 'orientation': 'vertical'},
                           'right_motor': pn.widgets.FloatSlider,  # {'widget_type': pn.widgets.FloatSlider, 'orientation': 'vertical'}
                           })

row = pn.Row(p, motors, pn.Column(simulator.param.agent_idx, simulator.agent_config), pn.Column(button, simulator.simulation_config))
row.servable()

def update_plot():
    if len(cds.selected.indices) > 0 and cds.selected.indices[0] != simulator.agent_idx:
        simulator.agent_idx = cds.selected.indices[0]
        simulator.pull_agent_config()
    state = simulator.get_nve_state()
    update_cds(state)


pcb_plot = pn.state.add_periodic_callback(update_plot, 10)


