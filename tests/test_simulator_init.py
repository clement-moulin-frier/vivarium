from vivarium.simulator.simulator import Simulator
from vivarium.environments.braitenberg.selective_sensing import init_state, SelectiveSensorsEnv


def test_init_simulator_no_args():
    """ Test the initialization of state without arguments """
    state = init_state()
    env = SelectiveSensorsEnv(state=state)
    simulator = Simulator(env_state=state, env=env)

    assert simulator


# TODO Remove
# def test_init_simulator_helper_fns():
#     """ Test the initialization of state without arguments """
#     simulator_state = init_simulator_state()
#     agents_state = init_agent_state(simulator_state=simulator_state)
#     objects_state = init_object_state(simulator_state=simulator_state)
#     entity_state = init_entity_state(simulator_state=simulator_state)

#     state = _init_state(
#         simulator_state=simulator_state,
#         agents_state=agents_state,
#         objects_state=objects_state,
#         entity_state=entity_state
#         )

#     simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid)

#     assert simulator


# TODO : Replace with functions from Selective Sensing Env 
# def test_init_simulator_args():
#     """ Test the initialization of state with arguments """
#     box_size = 100.0
#     max_agents = 10
#     max_objects = 2
#     col_eps = 0.1
#     col_alpha = 0.5
#     diameter = 5.0
#     friction = 0.1
#     behavior = 1
#     wheel_diameter = 2.0
#     speed_mul = 1.0
#     max_speed = 10.0
#     theta_mul = 1.0
#     prox_dist_max = 20.0
#     prox_cos_min = 0.0
#     color = "red"

#     simulator_state = init_simulator_state(
#         box_size=box_size,
#         max_agents=max_agents,
#         max_objects=max_objects,
#         collision_eps=col_eps,
#         collision_alpha=col_alpha)

#     entity_state = init_entity_state(
#         simulator_state,
#         diameter=diameter,
#         friction=friction)

#     agent_state = init_agent_state(
#         simulator_state,
#         behavior=behavior,
#         wheel_diameter=wheel_diameter,
#         speed_mul=speed_mul,
#         max_speed=max_speed,
#         theta_mul=theta_mul, 
#         prox_dist_max=prox_dist_max,
#         prox_cos_min=prox_cos_min)

#     object_state = init_object_state(
#         simulator_state,
#         color=color)

#     state = _init_state(
#         simulator_state=simulator_state,
#         agents_state=agent_state,
#         objects_state=object_state,
#         entity_state=entity_state)

#     simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid)

#     assert simulator