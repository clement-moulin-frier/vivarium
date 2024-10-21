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

N_DIGITS = 3


class Entity:
    """Entity class that represents an entity in the simulation
    """
    def __init__(self, config):
        self.config = config
        self._routines = {}
        self.user_events = {}

    def __getattr__(self, item):
        """Get the attribute of the entity

        :param item: item
        :return: attribute
        """
        if item in self.config.param_names():
            return getattr(self.config, item)
        else:
            return super().__getattr__(item)

    # TODO : Add a check to ensure that the attribute's value is authorized (according to params bounds)
    def __setattr__(self, item, val):
        """Set the attribute of the entity

        :param item: field to set
        :param val: value to set
        :return: attribute
        """
        if item != 'config' and item in self.config.param_names():
            self.user_events[item] = val # ensures the event is set during the run loop
        else:
            return super().__setattr__(item, val)
        
    def set_events(self):
        """Set the user events of the entity (events that are set during the run loop from the user)
        """
        for k, v in self.user_events.items():
            setattr(self.config, k, v)
        self.user_events = {}

    def attach_routine(self, routine_fn, name=None, interval=1):
        """Attach a routine to the entity

        :param routine_fn: routine_fn
        :param name: routine name, defaults to None
        :param interval: routine execution interval, defaults to 1
        """
        self._routines[name or routine_fn.__name__] = (routine_fn, interval)

    def detach_routine(self, name):
        """Detach a routine from the entity

        :param name: routine name
        """
        del self._routines[name]

    def detach_all_routines(self):
        """Detach all routines from the entity
        """
        self._routines = {}

    def routine_step(self, time):
        """Execute the entity's routines with their corresponding execution intervals
        """
        to_remove = []
        for name, (fn, interval) in self._routines.items():
            if time % interval == 0:
                try:
                    fn(self)
                except Exception as e:
                    lg.error(f"Error while executing routine: {e}, removing routine {name}")
                    to_remove.append(name)
            
            for name in to_remove:
                del self._routines[name]

    def infos(self):
        """Print the entity's infos

        :return: entity's infos
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
    """Agent class that represents an agent in the simulation

    :param Entity: Entity
    """
    def __init__(self, config):
        super().__init__(config)
        self.etype = EntityType.AGENT

        self.behaviors = {}
        self.active_behaviors = {}
        self.eating_range = 10
        self.diet = []
        self.ate = False
        self.simulation_entities = None
        self.logger = Logger()
        self.set_manual()

    def set_manual(self):
        """Set the agent's behavior to manual
        """
        self.behavior = np.full(shape=self.behavior.shape, fill_value=Behaviors.MANUAL.value)
        self.stop_motors()

    def sensors(self, sensed_entities=None):
        """Return the sensors values of the agent

        :param sensed_entities: sensed_entities of the sensors, defaults to None
        :return: sensors values
        """
        left, right = self.config.left_prox, self.config.right_prox
        if sensed_entities is not None:
            sensed_type_left, sensed_type_right = self.prox_sensed_ent_type
            left = left if sensed_type_left in sensed_entities else 0
            right = right if sensed_type_right in sensed_entities else 0
        return [left, right]

    def sensed_entities(self):
        """Return the left and right sensed entities of the agent

        :return: sensed entities
        """
        left_idx, right_idx = self.prox_sensed_ent_idx
        return [
            self.simulation_entities[left_idx], 
            self.simulation_entities[right_idx]
        ]
    
    def sense_attributes(self, sensed_attribute, default_value=None):
        """Return the sensed attribute of the left and right sensed entities

        :param sensed_attribute: sensed_attribute
        :param default_value: default value if the attribute is not found, defaults to None
        :return: sensed attributes
        """
        left_ent, right_ent = self.sensed_entities()
        return (
            getattr(left_ent, sensed_attribute, default_value), 
            getattr(right_ent, sensed_attribute, default_value)
        )
    def attach_behavior(self, behavior_fn, name=None, weight=1.):
        """Attach a behavior to the agent with a given weight

        :param behavior_fn: behavior_fn
        :param name: name, defaults to None
        :param weight: weight, defaults to 1.
        """
        self.behaviors[name or behavior_fn.__name__] = (behavior_fn, weight)

    def detach_behavior(self, name, stop_motors=False):
        """Detach a behavior from the agent and stop the motors if needed

        :param name: name
        :param stop_motors: stop motors signal, defaults to False
        """
        n = name.__name__ if hasattr(name, "__name__") else name
        if n in self.behaviors:
            del self.behaviors[n]
        if n in self.active_behaviors:
            del self.active_behaviors[n]
        if stop_motors:
            self.stop_motors()

    def detach_all_behaviors(self, stop_motors=False):
        """Detach all behaviors from the agent and stop the motors if needed

        :param stop_motors: stop motors signal, defaults to False
        """
        self.behaviors = {}
        self.active_behaviors = {}
        if stop_motors:
            self.stop_motors()

    def start_behavior(self, name):
        """Start a behavior of the agent

        :param name: behavior name
        """
        n = name.__name__ if hasattr(name, "__name__") else name
        self.active_behaviors[n] = self.behaviors[n]

    def start_all_behaviors(self):
        """Start all behaviors of the agent
        """
        for n in self.behaviors:
            self.active_behaviors[n] = self.behaviors[n]

    def stop_behavior(self, name):
        """Stop a behavior of the agent

        :param name: behavior name
        """
        n = name.__name__ if hasattr(name, "__name__") else name
        del self.active_behaviors[n]

    def check_behaviors(self):
        """Print the behaviors and active behaviors of the agent
        """
        if len(self.behaviors) == 0:
            print("No behaviors attached")
        else:
            print(f"Available behaviors: {list(self.behaviors.keys())}")
            if len(self.active_behaviors) == 0:
                print("No active behaviors")
            else:
                print(f"active behaviors: {list(self.active_behaviors.keys())}")

    def behave(self):
        """Make the agent behave according to its active behaviors
        """
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
        """Stop the motors of the agent
        """
        self.left_motor = 0
        self.right_motor = 0

    def has_eaten(self):
        """Check if the agent has eaten

        :return: True if the agent has eaten, False otherwise
        """
        val = self.ate
        self.ate = False
        return val
    
    def add_log(self, log_field, data):
        """Add a log to the agent's logger (e.g robot.add_log("left_prox", left_prox_value))

        :param log_field: log_field of the log
        :param data: data logged
        """
        self.logger.add(log_field, data)

    def get_log(self, log_field):
        """Get the log of the agent's logger for a specific log_field

        :param log_field: desired log_field
        :return: associated log_field data
        """
        return self.logger.get_log(log_field)

    def clear_all_logs(self):
        """Clear all logs of the agent's logger

        :return: cleared logs
        """
        return self.logger.clear()

    def infos(self, full_infos=False):
        """Print the agent's infos

        :param full_infos: full_infos, defaults to False
        :return: agent's infos
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
    """Object class that represents an object in the simulation

    :param Entity: Entity
    """
    def __init__(self, config):
        super().__init__(config)
        self.etype = EntityType.OBJECT


etype_to_class = {
    EntityType.AGENT: Agent, 
    EntityType.OBJECT: Object
}


class NotebookController(SimulatorController):
    """NotebookController class that enables the user to control the simulation from a notebook

    :param SimulatorController: SimulatorController
    """
    def __init__(self, **params):
        super().__init__(**params)
        self.all_entities = []
        # add agents and objects to the global entities list
        for etype in list(EntityType):
            setattr(self, f'{etype.name.lower()}s', [etype_to_class[etype](c) for c in self.configs[etype.to_state_type()]])
            self.all_entities.extend(getattr(self, f'{etype.name.lower()}s'))
        self.from_stream = True
        self.configs[StateType.SIMULATOR][0].freq = -1
        self.box_size = self.configs[StateType.SIMULATOR][0].box_size
        self.stop_apparition_flag = False
        self._is_running = False
        self._routines = {}
        self.set_all_user_events()
        self.update_agents_entities_list()
        # TODO :associate the entities subtypes ids with their actual names

    def update_agents_entities_list(self):
        """Give all agents the list of all entities in the simulation (usefull for advanced cases like eating)
        """
        for agent in self.agents:
            agent.simulation_entities = self.all_entities

    def spawn_entity(self, entity_idx, position=None):
        """Spawn an entity at a given position

        :param entity_idx: entity_idx
        :param position: position, defaults to None
        """
        entity = self.all_entities[entity_idx]
        try:
            if entity.exists:
                lg.warning(f"Entity {entity_idx} already exists")
                return
            if position is not None:
                # TODO : rounding doesn't prevent from error 
                entity.x_position = float(position[0])
                entity.y_position = float(position[1])
            entity.exists = True
            lg.info(f"Entity {entity_idx} spawned at {entity.x_position, entity.y_position}")
        except Exception as e:
            lg.error(f"Error while spawning entity {entity_idx}: {e}")

    def remove_entity(self, entity_idx):
        """Remove an entity

        :param entity_idx: entity_idx
        """
        entity = self.all_entities[entity_idx]
        if not entity.exists:
            lg.warning(f"Entity {entity_idx} already removed")
        entity.exists = False

    def remove_all_entities(self, entity_type):
        """Remove all entities of a given type

        :param entity_type: entity_type
        """
        for ent in self.all_entities:
            if ent.etype == entity_type:
                self.remove_entity(ent.idx)

    def periodic_entity_apparition(self, period, entity_type=None, position_range=None):
        """Spawn entities of type entity_type every period seconds within a given position range

        :param period: period
        :param entity_type: entity_type, defaults to None
        :param position_range: position_range, defaults to None
        """
        assert entity_type is not None, "Please specify the entity type"
        # transform the position range if not specified
        if position_range is None:
            box_size = self.state.simulator_state.box_size[0]
            position_range = ((0, box_size), (0, box_size)) 
        while not self.stop_apparition_flag:
            non_existing_ent_list = [ent.idx for ent in self.all_entities if not ent.exists and ent.subtype == entity_type]
            lg.debug(f"{non_existing_ent_list = }")
            if non_existing_ent_list:
                ent_idx = np.random.choice(non_existing_ent_list)
                x = np.random.uniform(position_range[0][0], position_range[0][1], 1)
                y = np.random.uniform(position_range[1][0], position_range[1][1], 1)
                self.spawn_entity(ent_idx, position=(x, y))
            else:
                lg.info(f'All entities of type {entity_type} are spawned')
            time.sleep(period)

    def start_entity_apparition(self, period=5, entity_type=None, position_range=None):
        """Start the apparition process for entities of type entity_type every period seconds

        :param period: period, defaults to 5
        :param entity_type: entity_type, defaults to None
        :param position_range: position range where entities can spawn, defaults to None
        """
        self.stop_apparition_flag = False
        thread = threading.Thread(target=self.periodic_entity_apparition, args=(period, entity_type, position_range))
        thread.start()

    # TODO : fix the hardcoded ressources id --> need the entities subtypes from server
    def start_ressources_apparition(self, period=5, position_range=None):
        """Start the ressources apparition process

        :param period: period, defaults to 5
        :param position_range: position_range, defaults to None
        """
        ressources_id = 1
        self.start_entity_apparition(period, entity_type=ressources_id, position_range=position_range)
        # attach the eating ressources routine
        self._routines[eating.__name__] = eating
                
    def stop_entity_apparition(self):
        """Stop any entities apparition process
        """
        self.stop_apparition_flag = True

    def set_all_user_events(self):
        """Set all user events from clients (interface or notebooks) for all entities
        """
        for e in self.all_entities:
            e.set_events()

    def run(self, threaded=True, num_steps=math.inf):
        """Run the simulation

        :param threaded: wether to run the simulation in a thread or not, defaults to True
        :param num_steps: num_steps, defaults to math.inf
        :raises RuntimeError: if the simulator is already started
        """
        if self._is_running:
            raise RuntimeError("Simulator is already started")
        self._is_running = True
        if threaded:
            threading.Thread(target=self._run).start()
        else:
            self._run(num_steps)

    def _run(self, num_steps=math.inf):
        """run the simulation for a given number of steps

        :param num_steps: num_steps, defaults to math.inf
        """
        t = 0
        while t < num_steps and self._is_running:
            with self.batch_set_state():
                # execute routines
                self.execute_simulator_routines()
                for entity in self.all_entities:
                    entity.routine_step(t)
                    entity.set_events()
                    if entity.etype == EntityType.AGENT:
                        entity.behave()
            self.state = self.client.step()
            self.pull_configs()
            t += 1
        self.stop()

    def stop(self):
        """Stop the simulation
        """
        self._is_running = False

    def wait(self, seconds):
        """Wait for a given number of seconds

        :param seconds: seconds
        """
        time.sleep(seconds)

    def execute_simulator_routines(self):
        """Execute the simulator routines
        """
        for fn in self._routines.values():
            fn(self)


def eating(controller):
    """make agents of the simulation eating the entities in their diet

    :param controller: NotebookController
    """
    for agent in controller.agents:
        # skip to next agent if the agent does not exist
        if not agent.exists:
            continue
        for object_type in agent.diet:
            ressources_idx = [ent.idx for ent in controller.all_entities if ent.subtype == object_type]
            distances = agent.config.proximity_map_dist[ressources_idx]
            in_range = distances < agent.eating_range
            for ress_idx, ent_idx in enumerate(ressources_idx):
                if in_range[ress_idx] and controller.all_entities[ent_idx].exists:
                    controller.remove_entity(ent_idx) 
                    agent.ate = True


# Logger class from pyvrep epuck (see if need to modify it)
class Logger(object):
    def __init__(self):
        """Logger class that logs data for the agents
        """
        self.logs = {}

    def add(self, log_field, data):
        """Add data to the log_field of the logger

        :param log_field: log_field
        :param data: data
        """
        if log_field not in self.logs:
            self.logs[log_field] = [data]
        else:
            self.logs[log_field].append(data)

    def get_log(self, log_field):
        """Get the log of the logger for a specific log_field

        :param log_field: log_field
        :return: data associated with the log_field
        """
        if log_field not in self.logs:
            print("No data in " + log_field)
            return []
        else:
            return self.logs[log_field]
        
    def clear(self):
        """Clear all logs of the logger
        """
        del self.logs
        self.logs = {}