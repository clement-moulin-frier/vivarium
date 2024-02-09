import numpy as np


from vivarium.simulator.simulator import Simulator
from vivarium.simulator.sim_computation import dynamics_rigid, StateType
from vivarium.controllers.config import AgentConfig, ObjectConfig, SimulatorConfig
from vivarium.controllers import converters
import vivarium.simulator.behaviors as behaviors


def test_simulator():
    simulator_config = SimulatorConfig(to_jit=True)

    agent_configs = [AgentConfig(idx=i,
                                 x_position=np.random.rand() * simulator_config.box_size,
                                 y_position=np.random.rand() * simulator_config.box_size,
                                 orientation=np.random.rand() * 2. * np.pi)
                     for i in range(simulator_config.n_agents)]

    object_configs = [ObjectConfig(idx=simulator_config.n_agents + i,
                                   x_position=np.random.rand() * simulator_config.box_size,
                                   y_position=np.random.rand() * simulator_config.box_size,
                                   orientation=np.random.rand() * 2. * np.pi)
                      for i in range(simulator_config.n_objects)]

    state = converters.set_state_from_config_dict({StateType.AGENT: agent_configs,
                                                   StateType.OBJECT: object_configs,
                                                   StateType.SIMULATOR: [simulator_config]
                                                   })

    simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid)

    simulator.run(threaded=False, num_loops=10)
