from vivarium.simulator import behaviors
from vivarium.simulator.states import init_simulator_state
from vivarium.simulator.states import init_agent_state
from vivarium.simulator.states import init_object_state
from vivarium.simulator.states import init_entities_state
from vivarium.simulator.states import init_state
from vivarium.simulator.simulator import Simulator
from vivarium.simulator.sim_computation import dynamics_rigid

NUM_STEPS = 50

def test_simulator_run():
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

    simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid)

    simulator.run(threaded=False, num_steps=NUM_STEPS)

    assert simulator