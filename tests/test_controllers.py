from math import pi
from random import random

import param

from vivarium.simulator.sim_computation import StateType
from vivarium.simulator.grpc_server.simulator_client import SimulatorGRPCClient
from vivarium.controllers.simulator_controller import SimulatorController
from vivarium.controllers.panel_controller import PanelController
from vivarium.controllers.notebook_controller import NotebookController

param.Dynamic.time_dependent = True

# Need to activate a server client to test these functions : 
# can simulate the behavior of a client-server connection to test the functions individually in the meantime

# def test_simulator_controller():
#     controller = SimulatorController()
#     controller.configs[StateType.AGENT][2].x_position = 1.
#     print(controller.client.get_state())

#     print('Done')


# def test_panel_controller():
#     simulator = PanelController(client=SimulatorGRPCClient())
#     print('Done')


# def test_notebook_controller():
#     controller = NotebookController()
#     c = controller.configs[StateType.AGENT][0]
#     with controller.batch_set_state():
#         for stype in list(StateType):
#             for c in controller.configs[stype]:
#                 for p in c.param_names():
#                     if p != 'idx':
#                         c.param.trigger(p)


#     objs = [controller.objects[0], controller.objects[1]]
#     with controller.batch_set_state():
#         for obj in objs:
#             obj.x_position = random() * controller.configs[StateType.SIMULATOR][0].box_size
#             obj.y_position = random() * controller.configs[StateType.SIMULATOR][0].box_size
#             obj.color = 'grey'
#             obj.orientation = random() * 2. * pi

#     print('Done')
 