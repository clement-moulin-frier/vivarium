import time
import math
import threading

import numpy as np
import logging

from vivarium.environments.braitenberg.simple import Behaviors
from vivarium.controllers.simulator_controller import SimulatorController
from vivarium.simulator.simulator_states import StateType, EntityType

lg = logging.getLogger(__name__)
    
if logging.root.handlers:
    lg.setLevel(logging.root.level)
else:
    lg.setLevel(logging.WARNING)


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

    # TODO : Add a check to ensure that the attribute's value is authorized (according to params bounds)
    def __setattr__(self, item, val):
        if item != 'config' and item in self.config.param_names():
            self.user_events[item] = val # ensures the event is set during the run loop
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

    def infos(self):
        """
        Returns a string that provides a detailed overview of the agent's key attributes.
        """
        dict_infos = self.config.to_dict()

        info_lines = []
        info_lines.append("Entity Overview:")
        info_lines.append(f"{'-' * 20}")
        info_lines.append(f"Entity Type: {self.etype.name}")
        info_lines.append(f"Entity Idx: {self.idx}")

        # Position
        info_lines.append(f"Position: x={dict_infos['x_position']:.2f}, y={dict_infos['y_position']:.2f}")
        info_lines.append("")
        
        return print("\n".join(info_lines))
    

class Agent(Entity):
    def __init__(self, config):
        super().__init__(config)
        self.etype = EntityType.AGENT

        self.behaviors = {}
        self.active_behaviors = {}
        self.can_eat = False
        self.eating_range = 10
        self.diet = []
        self.ate = False
        self.simulation_entities = None
        self.set_manual()

    def set_manual(self):
        self.behavior = np.full(shape=self.behavior.shape, fill_value=Behaviors.MANUAL.value)
        self.stop_motors()

    def sensors(self, sensed_entities=None):
        left, right = self.config.left_prox, self.config.right_prox
        if sensed_entities is not None:
            sensed_type_left, sensed_type_right = self.prox_sensed_ent_type
            left = left if sensed_type_left in sensed_entities else 0
            right = right if sensed_type_right in sensed_entities else 0
        return [left, right]

    # temporary method to return the sensed entities
    def sensed_entities(self):
        left_idx, right_idx = self.prox_sensed_ent_idx
        return [
            self.simulation_entities[left_idx], 
            self.simulation_entities[right_idx]
        ]
    
    def sense_attributes(self, sensed_attribute, default_value=None):
        left_ent, right_ent = self.sensed_entities()
        return (
            getattr(left_ent, sensed_attribute, default_value), 
            getattr(right_ent, sensed_attribute, default_value)
        )
    def attach_behavior(self, behavior_fn, name=None, weight=1.):
        self.behaviors[name or behavior_fn.__name__] = (behavior_fn, weight)

    def detach_behavior(self, name, stop_motors=False):
        n = name.__name__ if hasattr(name, "__name__") else name
        if n in self.behaviors:
            del self.behaviors[n]
        if n in self.active_behaviors:
            del self.active_behaviors[n]
        if stop_motors:
            self.stop_motors()

    def detach_all_behaviors(self, stop_motors=False):
        self.behaviors = {}
        self.active_behaviors = {}
        if stop_motors:
            self.stop_motors()

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
                    # problem here 
                    total_motor += w * np.array(fn(self))
                    total_weights += w
                motors = total_motor / total_weights
            except Exception as e:
                lg.error(f"Error while computing motor values: {e}")
                motors = np.zeros(2)
        self.left_motor, self.right_motor = motors

    def stop_motors(self):
        self.left_motor = 0
        self.right_motor = 0

    def has_eaten(self):
        val = self.ate
        self.ate = False
        return val

    def infos(self, full_infos=False):
        """
        Returns a string that provides a detailed overview of the agent's key attributes.
        """
        super().infos()
        info_lines = []
        sensors = self.sensors()
        info_lines.append(f"Sensors: Left={sensors[0]:.2f}, Right={sensors[1]:.2f}")
        info_lines.append(f"Motors: Left={self.left_motor:.2f}, Right={self.right_motor:.2f}")
        
        # List behaviors
        if self.behaviors:
            info_lines.append("Behaviors:")
            for name, (behavior_fn, weight) in self.behaviors.items():
                info_lines.append(f"  - {name}: Function={behavior_fn.__name__}, Weight={weight}")
        else:
            info_lines.append("Behaviors: None")
        # TODO : might add active behaviors here as well

        # See if we print that by default
        info_lines.append('') # add a space between other infos and eating infos atm
        info_lines.append(f"Can eat: {self.can_eat}")
        info_lines.append(f"Diet: {self.diet}")
        info_lines.append(f"Eating range: {self.eating_range}")
        
        dict_infos = self.config.to_dict()
        if full_infos:

            info_lines.append("\nConfiguration Details:")
            for k, v in dict_infos.items():
                if k not in ['x_position', 'y_position', 'behavior', 'left_motor', 'right_motor', 'params', 'sensed']:
                    info_lines.append(f"  - {k}: {v}")

        info_lines.append("")
        
        return print("\n".join(info_lines))


class Object(Entity):
    def __init__(self, config):
        super().__init__(config)
        self.etype = EntityType.OBJECT


etype_to_class = {
    EntityType.AGENT: Agent, 
    EntityType.OBJECT: Object
}


class NotebookController(SimulatorController):
    def __init__(self, **params):
        super().__init__(**params)
        self.all_entities = []
        # add agents and objects to the global entities list
        for etype in list(EntityType):
            setattr(self, f'{etype.name.lower()}s', [etype_to_class[etype](c) for c in self.configs[etype.to_state_type()]])
            self.all_entities.extend(getattr(self, f'{etype.name.lower()}s'))
        self.from_stream = True
        self.configs[StateType.SIMULATOR][0].freq = -1
        self.stop_apparition_flag = False
        self._is_running = False
        self._routines = {}
        self.set_all_user_events()
        self.update_agents_entities_list()
        # TODO :associate the entities subtypes ids with their actual names

    def update_agents_entities_list(self):
        for agent in self.agents:
            agent.simulation_entities = self.all_entities

    def spawn_entity(self, entity_idx, position=None):
        entity = self.all_entities[entity_idx]
        if entity.exists:
            lg.warning(f"Entity {entity_idx} already exists")
        if position is not None:
            # TODO : test updating the config first
            # configs[StateType.AGENT][2].x_position = 1.
            entity.x_position = position[0]
            entity.y_position = position[1]
        entity.exists = True
        lg.info(f"Entity {entity_idx} spawned at {entity.x_position, entity.y_position}")

    def remove_entity(self, entity_idx):
        entity = self.all_entities[entity_idx]
        if not entity.exists:
            lg.warning(f"Entity {entity_idx} already removed")
        entity.exists = False

    def remove_all_entities(self, entity_type):
        for ent in self.all_entities:
            if ent.etype == entity_type:
                self.remove_entity(ent.idx)

    def periodic_entity_apparition(self, period, entity_type=None, position_range=None):
        assert entity_type is not None, "Please specify the entity type"
        # transform the position range if not specified
        if position_range is None:
            box_size = self.state.simulator_state.box_size[0]
            position_range = ((0, box_size), (0, box_size)) 
        while not self.stop_apparition_flag:
            non_existing_ent_list = [ent.idx for ent in self.all_entities if not ent.exists and ent.subtype == entity_type]
            lg.debug(f"{non_existing_ent_list = }")
            if non_existing_ent_list:
                # there are entities of this type that are not spawned
                idx = np.random.choice(non_existing_ent_list)
                x = np.random.uniform(position_range[0][0], position_range[0][1], 1)
                y = np.random.uniform(position_range[1][0], position_range[1][1], 1)
                # self.spawn_entity(idx, position=(x, y))
                # TODO : at the moment don't set random positions because hard problem to debug
                self.spawn_entity(idx, position=None)
            else:
                lg.info(f'All entities of type {entity_type} are spawned')
            time.sleep(period)

    def start_entity_apparition(self, period=5, entity_type=None, position_range=None):
        self.stop_apparition_flag = False
        thread = threading.Thread(target=self.periodic_entity_apparition, args=(period, entity_type, position_range))
        thread.start()
                
    def stop_entity_apparition(self):
        self.stop_apparition_flag = True

    def eat_ressource(self, agent):
        # Or could do if agent.diet is None
        if not agent.can_eat or not agent.exists:
            return
        # TODO : could optimize this and not recompute the ressources idx at each iteration
        for object_type in agent.diet:
            ressources_idx = [ent.idx for ent in self.all_entities if ent.subtype == object_type]
            distances = agent.config.proximity_map_dist[ressources_idx]
            in_range = distances < agent.eating_range
            for ress_idx, ent_idx in enumerate(ressources_idx):
                if in_range[ress_idx] and self.all_entities[ent_idx].exists:
                    self.remove_entity(ent_idx) 
                    agent.ate = True

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
                # execute routines
                self.execute_simulator_routines()
                for entity in self.all_entities:
                    entity.routine_step()
                    entity.set_events()
                    if entity.etype == EntityType.AGENT:
                        entity.behave()
                        # check how to do this
                        self.eat_ressource(entity)

            self.state = self.client.step()
            self.pull_configs()
            t += 1
        self.stop()

    def stop(self):
        self._is_running = False

    def wait(self, seconds):
        time.sleep(seconds)

    # TODO : clean redundancy
    def execute_simulator_routines(self):
        for fn in self._routines.values():
            fn(self)

