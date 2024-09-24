import time
import math
import threading

import numpy as np
import logging

from vivarium.environments.braitenberg.simple import Behaviors
from vivarium.controllers.simulator_controller import SimulatorController
from vivarium.simulator.simulator_states import StateType, EntityType

lg = logging.getLogger(__name__)


# TODO : Add documentation
class Entity:
    def __init__(self, config):
        self.config = config
        self._routines = {}
        self.user_events = {}

    def __getattr__(self, item):
        if item in self.config.param_names():
            return getattr(self.config, item)
        else:
            return super().__getattr__(item)

    def __setattr__(self, item, val):
        if item != 'config' and item in self.config.param_names():
            self.user_events[item] = val # ensures the event is set during the run loop
            # return setattr(self.config, item, val) # ensures the event is set if the simulator is not running
        else:
            return super().__setattr__(item, val)
        
    def set_events(self):
        for k, v in self.user_events.items():
            setattr(self.config, k, v)
        self.user_events = {}

    def attach_routine(self, routine_fn, name=None):
        self._routines[name or routine_fn.__name__] = routine_fn

    def detach_routine(self, name):
        del self._routines[name]

    def detach_all_routines(self):
        self._routines = {}

    def routine_step(self):
        for fn in self._routines.values():
            fn(self)


# TODO : Add documentation
class Agent(Entity):
    def __init__(self, config):
        super().__init__(config)
        self.etype = EntityType.AGENT

        self.behaviors = {}
        self.active_behaviors = {}
        self.set_manual()

    def set_manual(self):
        self.behavior = np.full(shape=self.behavior.shape, fill_value=Behaviors.MANUAL.value)
        self.stop_motors()

    def sensors(self, sensed_entities=None):
        left, right = self.config.left_prox, self.config.right_prox
        if sensed_entities is not None:
            sensed_type_left, sensed_type_right = self.prox_sensed_ent
            left = left if sensed_type_left in sensed_entities else 0
            right = right if sensed_type_right in sensed_entities else 0
        return [left, right]
    
    # TODO : 
    def sense_agents_attributes(self, sensed_agents_ids, sensed_attributes):
        pass
        

    def attach_behavior(self, behavior_fn, name=None, weight=1.):
        self.behaviors[name or behavior_fn.__name__] = (behavior_fn, weight)

    def detach_behavior(self, name):
        n = name.__name__ if hasattr(name, "__name__") else name
        if n in self.behaviors:
            del self.behaviors[n]
        if n in self.active_behaviors:
            del self.active_behaviors[n]

    def detach_all_behaviors(self):
        self.behaviors = {}
        self.active_behaviors = {}

    def start_behavior(self, name):
        n = name.__name__ if hasattr(name, "__name__") else name
        self.active_behaviors[n] = self.behaviors[n]

    def start_all_behaviors(self):
        for n in self.behaviors:
            self.active_behaviors[n] = self.behaviors[n]

    def stop_behavior(self, name):
        n = name.__name__ if hasattr(name, "__name__") else name
        del self.active_behaviors[n]

    def check_behaviors(self):
        if len(self.behaviors) == 0:
            print("No behaviors attached")
        else:
            print(f"Available behaviors: {list(self.behaviors.keys())}")
            if len(self.active_behaviors) == 0:
                print("No active behaviors")
            else:
                print(f"active behaviors: {list(self.active_behaviors.keys())}")

    def behave(self):
        if len(self.active_behaviors) == 0:
            return
        else:
            # add a try to prevent the simulator from crashing when a bad behavior function is attached
            try:
                total_weights = 0.
                total_motor = np.zeros(2)
                for fn, w in self.active_behaviors.values():
                    total_motor += w * np.array(fn(self))
                    total_weights += w
                motors = total_motor / total_weights
            except Exception as e:
                lg.error(f"Error while computing motor commands: {e}")
                motors = np.zeros(2)
        self.left_motor, self.right_motor = motors

    def stop_motors(self):
        self.left_motor = 0
        self.right_motor = 0

    def infos(self, full_infos=False):
        """
        Returns a string that provides a detailed overview of the agent's key attributes.
        """
        dict_infos = self.config.to_dict()

        info_lines = []
        info_lines.append("Agent Overview:")
        info_lines.append(f"{'-' * 20}")
        info_lines.append(f"Entity Type: {self.etype.name}")

        # Position
        info_lines.append(f"Position: x={dict_infos['x_position']:.2f}, y={dict_infos['y_position']:.2f}")
        
        # List behaviors
        if self.behaviors:
            info_lines.append("Behaviors:")
            for name, (behavior_fn, weight) in self.behaviors.items():
                info_lines.append(f"  - {name}: Function={behavior_fn.__name__}, Weight={weight}")
        else:
            info_lines.append("Behaviors: None")
        # Sensor configurations
        sensors = self.sensors()
        info_lines.append(f"Sensors: Left={sensors[0]:.2f}, Right={sensors[1]:.2f}")
        # Motor states
        info_lines.append(f"Motors: Left={self.left_motor:.2f}, Right={self.right_motor:.2f}")

        if full_infos:
            info_lines.append("\nConfiguration Details:")
            for k, v in dict_infos.items():
                if k not in ['x_position', 'y_position', 'behavior', 'left_motor', 'right_motor', 'params', 'sensed']:
                    info_lines.append(f"  - {k}: {v}")
        
        return print("\n".join(info_lines))


# TODO : Add documentation
class Object(Entity):
    def __init__(self, config):
        super().__init__(config)
        self.etype = EntityType.OBJECT


etype_to_class = {
    EntityType.AGENT: Agent, 
    EntityType.OBJECT: Object
}


# TODO : Add documentation
class NotebookController(SimulatorController):
    def __init__(self, **params):
        super().__init__(**params)
        self.all_entities = []
        for etype in list(EntityType):
            # set the attributes for agents and objects
            setattr(self, f'{etype.name.lower()}s', [etype_to_class[etype](c) for c in self.configs[etype.to_state_type()]])
            self.all_entities.extend(getattr(self, f'{etype.name.lower()}s'))
        self.from_stream = True
        self.configs[StateType.SIMULATOR][0].freq = -1
        self.set_all_user_events()
        self._is_running = False

    def set_all_user_events(self):
        for e in self.all_entities:
            e.set_events()

    def run(self, threaded=True, num_steps=math.inf):
        if self._is_running:
            raise RuntimeError("Simulator is already started")
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
                    e.set_events()
                for ag in self.agents:
                    ag.behave()
            self.state = self.client.step()
            self.pull_configs()
            t += 1
        self.stop()

    def start_food_apparition(self, period=5):
        pass

    def stop(self):
        self._is_running = False

    def wait(self, seconds):
        time.sleep(seconds)

