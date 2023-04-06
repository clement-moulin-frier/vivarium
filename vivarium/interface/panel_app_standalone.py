import numpy as np

from vivarium.simulator.rest_api import SimulatorRestClient
import vivarium.simulator.config as config

import panel as pn
import param

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Button, PointDrawTool, HoverTool, Range1d
from bokeh.layouts import layout
from bokeh.events import ButtonClick

def normal(array):
    normals = np.zeros((array.shape[0], 2))
    normals[:, 0] = np.cos(array)
    normals[:, 1] = np.sin(array)
    return normals

class Application:
    def __init__(self, simulator):
        self.simulator = simulator #  SimulatorRestClient()
    def init_plot(self):
        self.pn.extension()

        # sim_config = simulator.get_sim_config()
        # agent_config = simulator.get_agent_config()
        # state = simulator.get_state()

        box_size = self.simulator.simulation_config.box_size

        positions = np.array(self.simulator.state.positions)

        N = positions.shape[0]

        x = positions[:, 0]
        y = x = positions[:, 1]

        self.radius = self.simulator.agent_config.base_length / 2.
        colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)]

        thetas = np.array(self.simulator.state.thetas)
        normals = normal(thetas)

        orientation_lines_x = [[xx, xx + self.radius * n[0]] for xx, n in zip(x, normals)]
        orientation_lines_y = [[yy, yy + self.radius * n[1]] for yy, n in zip(y, normals)]

        self.cds = ColumnDataSource(data={'x': x, 'y': y,
                                     'ox': orientation_lines_x, 'oy': orientation_lines_y,
                                     'r': np.ones(N) * self.radius,
                                     'fc': colors
                                     }
                               )

        tools = "crosshair,pan,wheel_zoom,box_zoom,reset,tap,box_select,lasso_select"

        self.p = figure(tools=tools)
        self.p.axis.major_label_text_font_size = "24px"
        hover = HoverTool(tooltips=None, mode="vline")
        self.p.add_tools(hover)
        self.p.x_range = Range1d(0, box_size)
        self.p.y_range = Range1d(0, box_size)

        orientations = p.multi_line('ox', 'oy', source=self.cds, color='black', line_width=1)
        r = p.circle('x', 'y', radius='r',
                     fill_color='fc', fill_alpha=0.6, line_color=None,
                     hover_fill_color="black", hover_fill_alpha=0.7, hover_line_color=None, source=self.cds)

        self.button = Button(name="Start" if self.simulator.is_started else "Stop")
        def callback(event):
            if self.simulator.is_started:
                self.name = "Stop"
                self.simulator.stop()
            else:
                self.button.name = "Start"
                self.start()

        self.button.on_event(ButtonClick, callback)

        draw_tool = PointDrawTool(renderers=[r])
        self.p.add_tools(draw_tool)
        self.p.toolbar.active_tap = draw_tool

        lo = layout([[self.button], [self.p]])

        self.bk_pane = pn.pane.Bokeh(lo)

        self.bk_pane.servable()

        self.pcb = pn.state.add_periodic_callback(self.update_plot, 10)

    def update_plot(self):

        state = self.simulator.get_state()

        positions = np.array(self.simulator.state.positions)

        x = positions[:, 0]
        y = positions[:, 1]

        normals = normal(np.array(state['thetas']))

        orientation_lines_x = [[xx, xx + self.radius * n[0]] for xx, n in zip(x, normals)]
        orientation_lines_y = [[yy, yy + self.radius * n[1]] for yy, n in zip(y, normals)]

        self.cds.data['x'] = x
        self.cds.data['y'] = y
        self.cds.data['ox'] = orientation_lines_x
        self.cds.data['oy'] = orientation_lines_y

if __name__ == "__main__":
from vivarium.simulator import config
from vivarium.simulator.simulator import Simulator

    agent_config = config.AgentConfig()
    simulation_config = config.SimulatorConfig(agent_config=agent_config)
    population_config = config.PopulationConfig()
    behavior_config = config.BehaviorConfig(population_config=population_config)

    simulator = Simulator(simulation_config=simulation_config, agent_config=agent_config,
                          behavior_config=behavior_config, population_config=population_config)

    app = Application(simulator)

    app.init_plot()

