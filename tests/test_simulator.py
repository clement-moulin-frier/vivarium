import numpy as np

import vivarium.simulator.behaviors as behaviors
from vivarium.simulator.simulator import Simulator
from vivarium.simulator.sim_computation import dynamics_rigid, StateType
from vivarium.controllers.config import AgentConfig, ObjectConfig, SimulatorConfig
from vivarium.controllers import converters


NUM_LOOPS = 50


# First smoke test, we could split it into different parts later (initialization, run, ...)
def test_simulator_run():
    simulator_config = SimulatorConfig(to_jit=True)

    agent_configs = [
        AgentConfig(
            idx=i,
            x_position=np.random.rand() * simulator_config.box_size,
            y_position=np.random.rand() * simulator_config.box_size,
            orientation=np.random.rand() * 2. * np.pi)
        for i in range(simulator_config.n_agents)
        ]

    object_configs = [
        ObjectConfig(
            idx=simulator_config.n_agents + i,
            x_position=np.random.rand() * simulator_config.box_size,
            y_position=np.random.rand() * simulator_config.box_size,
            orientation=np.random.rand() * 2. * np.pi)
        for i in range(simulator_config.n_objects)
        ]

    state = converters.set_state_from_config_dict(
        {StateType.AGENT: agent_configs,
        StateType.OBJECT: object_configs,
        StateType.SIMULATOR: [simulator_config]
        }
        )

    col_params = {'eps': 0.3, 'alpha': 0.4, 'dist_mul': 1.1}
    simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid, col_params)

    simulator.run(threaded=False, num_loops=NUM_LOOPS)

    assert simulator