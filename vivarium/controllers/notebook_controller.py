import math
import threading

import numpy as np
import logging

from vivarium.controllers.simulator_controller import SimulatorController
from vivarium.simulator.sim_computation import StateType, EntityType

lg = logging.getLogger(__name__)

class Entity:
    def __init__(self, config):
        self.config = config
        self._routines = {}

    def __getattr__(self, item):
        if item in self.config.param_names():
            return getattr(self.config, item)
        else:
            return super().__getattr__(item)

    def __setattr__(self, item, val):
        if item != 'config' and item in self.config.param_names():
            return setattr(self.config, item, val)
        else:
            return super().__setattr__(item, val)

    def attach_routine(self, routine_fn, name=None):
        self._routines[name or routine_fn.__name__] = routine_fn

    def detach_routine(self, name):
        del self._routines[name]

    def detach_all_routines(self):
        self._routines = {}

    def routine_step(self):
        for fn in self._routines.values():
            fn(self)


class Agent(Entity):
    def __init__(self, config):
        super().__init__(config)
        self.config.behavior = 'manual'
        self.etype = EntityType.AGENT

        self.behaviors = {}

    def sensors(self):
        return [self.config.left_prox, self.config.right_prox]

    def attach_behavior(self, behavior_fn, name=None, weight=1.):
        self.behaviors[name or behavior_fn.__name__] = (behavior_fn, weight)

    def detach_behavior(self, name):
        del self.behaviors[name]

    def detach_all_behaviors(self):
        self.behaviors = {}

    def behave(self):
        if len(self.behaviors) == 0:
            motors = [0., 0.]
        else:
            total_weights = 0.
            total_motor = np.zeros(2)
            for fn, w in self.behaviors.values():
                total_motor += w * np.array(fn(self))
                total_weights += w
            motors = total_motor / total_weights
        self.left_motor, self.right_motor = motors


class Object(Entity):
    def __init__(self, config):
        super().__init__(config)
        self.etype = EntityType.OBJECT


etype_to_class = {EntityType.AGENT: Agent, EntityType.OBJECT: Object}


class NotebookController(SimulatorController):

    def __init__(self, **params):
        super().__init__(**params)
        self.all_entities = []
        for etype in list(EntityType):
            setattr(self, f'{etype.name.lower()}s', [etype_to_class[etype](c) for c in self.configs[etype.to_state_type()]])
            self.all_entities.extend(getattr(self, f'{etype.name.lower()}s'))
        self.from_stream = True
        self.configs[StateType.SIMULATOR][0].freq = -1
        self._is_running = False

    def run(self, threaded=False, num_steps=math.inf):
        if self.is_started():
            raise Exception("Simulator is already started")
        self._is_running = True
        if threaded:
            threading.Thread(target=self._run).start()
        else:
            self._run(num_steps)

    def _run(self, num_steps=math.inf):
        t = 0
        while t < num_steps and self._is_running:
            with self.batch_set_state():
                for e in self.all_entities:
                    e.routine_step()
                for ag in self.agents:
                    ag.behave()
            self.state = self.client.step()
            self.pull_configs()

            t += 1
        self._is_running = False

    def stop(self):
        self._is_running = False


if __name__ == "__main__":

    controller = NotebookController()
    c = controller.configs[StateType.AGENT][0]
    with controller.batch_set_state():
        for stype in list(StateType):
            for c in controller.configs[stype]:
                for p in c.param_names():
                    if p != 'idx':
                        c.param.trigger(p)

    from random import random
    from math import pi

    objs = [controller.objects[0], controller.objects[1]]
    with controller.batch_set_state():
        for obj in objs:
            obj.x_position = random() * controller.configs[StateType.SIMULATOR][0].box_size
            obj.y_position = random() * controller.configs[StateType.SIMULATOR][0].box_size
            obj.color = 'grey'
            obj.orientation = random() * 2. * pi

    lg.info('Done')
