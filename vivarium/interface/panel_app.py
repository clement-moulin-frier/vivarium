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


def update_selected(*events):
    cds.selected.indices = simulator.selected_agents


pn.extension()

simulator = SimulatorController(client=SimulatorGRPCClient())

simulator.param.watch(update_selected, ['selected_agents'], onlychanged=True)

state = simulator.get_nve_state()

max_agents = 1000
all_colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(np.random.rand(max_agents) * 200 + 50, np.random.rand(max_agents) * 200 + 50)]


def get_cds_data(state):
    pos = state.position.center
    x, y = pos[:, 0], pos[:, 1]
    thetas = state.position.orientation
    radii = state.base_length / 2.
    colors = state.color  # ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)]

    normals = normal(np.array(thetas))

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

add_agent_text_pre = pn.widgets.StaticText(name='Pre text', value='Add ')
n_new_agents = pn.widgets.IntInput(name='N new agents', value=0, step=1, start=0, end=1000)
add_agent_text_post = pn.widgets.StaticText(name='Pre text', value=' agents')
add_agent_button = pn.widgets.Button(name='Add agents')


def add_agents(event):
    simulator.add_agents(n_new_agents.value)

add_agent_button.on_click(add_agents)

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
sim_panel = pn.Param(simulator.param)
                  #    ,
                  # widgets={'left_motor': pn.widgets.FloatSlider,  # {'widget_type': pn.widgets.FloatSlider, 'orientation': 'vertical'},
                  #          'right_motor': pn.widgets.FloatSlider,  # {'widget_type': pn.widgets.FloatSlider, 'orientation': 'vertical'}
                  #          })

row = pn.Row(pn.Column(button, p, pn.Row(add_agent_text_pre, n_new_agents,add_agent_text_post, add_agent_button)), sim_panel, simulator.agent_config)  # , pn.Column(simulator.param.selected_agents, simulator.agent_config), pn.Column(button, simulator.simulation_config))
row.servable()

def update_plot():
    print(simulator.selected_agents)
    if len(cds.selected.indices) > 0 and cds.selected.indices != simulator.selected_agents:
        simulator.selected_agents = cds.selected.indices
        simulator.pull_agent_config()
    state = simulator.get_nve_state()
    update_cds(state)


pcb_plot = pn.state.add_periodic_callback(update_plot, 10)


