from vivarium.simulator import behaviors
from vivarium.simulator.states import init_state
from vivarium.simulator.simulator import Simulator
from vivarium.simulator.physics_engine import dynamics_rigid

NUM_STEPS = 50

def test_simulator_run():
    state = init_state()
    simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid)
    simulator.run(threaded=False, num_steps=NUM_STEPS)

    assert simulator