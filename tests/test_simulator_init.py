from vivarium.simulator import behaviors
from vivarium.simulator.states import init_simulator_state
from vivarium.simulator.states import init_agent_state
from vivarium.simulator.states import init_object_state
from vivarium.simulator.states import init_entities_state
from vivarium.simulator.states import init_state
from vivarium.simulator.simulator import Simulator
from vivarium.simulator.sim_computation import dynamics_rigid


def test_init_simulator_no_args():
    """ Test the initialization of state without arguments """
    simulator_state = init_simulator_state()
    agents_state = init_agent_state(simulator_state=simulator_state)
    objects_state = init_object_state(simulator_state=simulator_state)
    entities_state = init_entities_state(simulator_state=simulator_state)

    state = init_state(
        simulator_state=simulator_state,
        agents_state=agents_state,
        objects_state=objects_state,
        entities_state=entities_state
        )

    simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid)

    assert simulator


def test_init_simulator_args():
    """ Test the initialization of state with arguments """
    box_size = 100.0
    max_agents = 10
    max_objects = 2
    col_eps = 0.1
    col_alpha = 0.5

    diameter = 5.0
    friction = 0.1
    behavior = 1
    wheel_diameter = 2.0
    speed_mul = 1.0
    theta_mul = 1.0
    prox_dist_max = 20.0
    prox_cos_min = 0.0
    color = "red"

    simulator_state = init_simulator_state(
        box_size=box_size,
        max_agents=max_agents,
        max_objects=max_objects,
        collision_eps=col_eps,
        collision_alpha=col_alpha)

    entities_state = init_entities_state(
        simulator_state,
        diameter=diameter,
        friction=friction)

    agent_state = init_agent_state(
        simulator_state,
        behavior=behavior,
        wheel_diameter=wheel_diameter,
        speed_mul=speed_mul,
        theta_mul=theta_mul, 
        prox_dist_max=prox_dist_max,
        prox_cos_min=prox_cos_min)

    object_state = init_object_state(
        simulator_state,
        color=color)

    state = init_state(
        simulator_state=simulator_state,
        agents_state=agent_state,
        objects_state=object_state,
        entities_state=entities_state)

    simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid)

    assert simulator