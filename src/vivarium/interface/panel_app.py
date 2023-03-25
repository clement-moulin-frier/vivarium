import time

import numpy as np

from bokeh.io import push_notebook, show, output_notebook
from bokeh.models import HoverTool, Range1d
from bokeh.plotting import figure

from urllib.parse import urljoin
import requests


import panel as pn

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

pn.extension()

sim_server_url = 'http://localhost:8086'

def get_sim_config(sim_server_url=sim_server_url):
    sim_config = requests.get(urljoin(sim_server_url, 'get_sim_config'))
    return sim_config.json()

def get_sim_state(sim_server_url=sim_server_url):
    state = requests.post(urljoin(sim_server_url, 'get_state'))
    return state.json()

def run():
    requests.get(urljoin(sim_server_url, 'run'))

sim_config = get_sim_config()
state = get_sim_state()

box_size = sim_config['box_size']
map_dim = sim_config['map_dim']

positions = np.array(state['PREY']['positions'])

N = positions.shape[0]

x = positions[:, 0]
y = x = positions[:, 1]

radius = sim_config['base_lenght'] / 2.
colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)]

#orientation_lines_x = [[xx, xx+radius] for xx in x]
#orientation_lines_y = [[yy, yy] for yy in y]

#x = cds.data['x']
#y = cds.data['y']

def normal(array):
    normals = np.zeros((array.shape[0], map_dim))
    normals[:, 0] = np.cos(array)
    normals[:, 1] = np.sin(array)
    return normals

thetas = np.array(state['PREY']['thetas'])
normals = normal(thetas)
#normals = np.array(normal(state['PREY']['thetas']))

orientation_lines_x = [[xx, xx + radius * n[0]] for xx, n in zip(x, normals)]
orientation_lines_y = [[yy, yy + radius * n[1]] for yy, n in zip(y, normals)]

#positions = jnp.vstack((jnp.array(x), jnp.array(y))).T

cds = ColumnDataSource(data={'x': x, 'y': y,
                             'ox': orientation_lines_x, 'oy': orientation_lines_y,
                             'r': np.ones(N) * radius,
                             'fc': colors
                            })

# In[23]:


TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,tap,box_select,lasso_select"

p = figure(tools=TOOLS)
p.axis.major_label_text_font_size = "24px"
hover = HoverTool(tooltips=None, mode="vline")
p.add_tools(hover)
p.x_range = Range1d(0, box_size)
p.y_range = Range1d(0, box_size)

orientations = p.multi_line('ox', 'oy', source =cds, color='black', line_width=1)
r = p.circle('x', 'y', radius='r',
             fill_color='fc', fill_alpha=0.6, line_color=None,
             hover_fill_color="black", hover_fill_alpha=0.7, hover_line_color=None, source=cds)



from bokeh.events import ButtonClick
from bokeh.models import Button, PointDrawTool

button = Button()

global sim_start
sim_start = False

def callback(event):
    global sim_start
    if sim_start:
        sim_start = False
    else:
        sim_start = True
    #print(state['PREY'].positions.at[0,:].set(jnp.array([50., 50.])))

button.on_event(ButtonClick, callback)

draw_tool = PointDrawTool(renderers=[r])
p.add_tools(draw_tool)
p.toolbar.active_tap = draw_tool


# In[ ]:


from bokeh.layouts import layout
lo = layout([[button], [p]])


# In[24]:


bk_pane = pn.pane.Bokeh(lo)

bk_pane.servable()



def update_plot():
    global sim_start
    #global state, neighbors, sim_start

    # if not sim_start:
    #     print('stop')
    #     x = cds.data['x']
    #     y = cds.data['y']
    #
    #     normals = normal(np.array(normal(state['PREY']['thetas'])))
    #
    #     orientation_lines_x = [[xx, xx + radius * n[0]] for xx, n in zip(x, normals)]
    #     orientation_lines_y = [[yy, yy + radius * n[1]] for yy, n in zip(y, normals)]
    #
    #     cds.data['ox'] = orientation_lines_x
    #     cds.data['oy'] = orientation_lines_y
    #
    #     #positions = jnp.vstack((jnp.array(x), jnp.array(y))).T
    #
    #     #state['PREY'] = Population(positions, state['PREY'].thetas, state['PREY'].entity_type)
    #
    #     return

    #print('start')
    #run()
    state = get_sim_state()

    positions = np.array(state['PREY']['positions'])
    #print(positions)
    #new_state, neighbors = lax.fori_loop(0, 50, update, (state, neighbors))
    # if pos_change:
    #     print(pos_change)
    #     new_state['PREY'].positions = jnp.zeros((N, 2))
    #     #new_state['PREY'].positions[0, 0] = 50. # new_state['PREY'].positions.at[(0,slice(None))].set(jnp.array([50., 50.]))
    #     print(new_state['PREY'].positions)
    #     pos_change = False
    #print(np.array(state['PREY'].thetas[0]))
    x = positions[:, 0]
    y = positions[:, 1]

    normals = normal(np.array(state['PREY']['thetas']))

    orientation_lines_x = [[xx, xx + radius * n[0]] for xx, n in zip(x, normals)]
    orientation_lines_y = [[yy, yy + radius * n[1]] for yy, n in zip(y, normals)]

    cds.data['x'] = x
    cds.data['y'] = y
    cds.data['ox'] = orientation_lines_x
    cds.data['oy'] = orientation_lines_y

    #state = new_state

    #if pos_change:
    #    state['PREY'] = Population(state['PREY'].positions.at[0,:].set(50.), state['PREY'].thetas, state['PREY'].entity_type)
    #    pos_change = False

    #pn.io.push_notebook(bk_pane) # Only needed when running in notebook context

pcb = pn.state.add_periodic_callback(update_plot, 10)

