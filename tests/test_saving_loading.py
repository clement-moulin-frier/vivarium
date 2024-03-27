# !!! TODO : currently the file isn't testing anything because there are problems while comparing jax objects
# !!! + numerical errors while saving and loading with pickle, a better option would be to directly use jax / flax options to save 

import logging

import numpy as np
import jax.numpy as jnp 

from vivarium.simulator import behaviors
from vivarium.simulator.sim_computation import dynamics_rigid, StateType
from vivarium.controllers.config import AgentConfig, ObjectConfig, SimulatorConfig
from vivarium.controllers import converters
from vivarium.simulator.simulator import Simulator

lg = logging.getLogger(__name__)

num_steps = 10
save = False
saving_name = ''
box_size = 100.0
n_agents = 10
n_objects = 2
num_steps_lax = 4
dt = 0.1
freq = 40.0
neighbor_radius = 100.0
to_jit = True
use_fori_loop = False


def test_saving_loading():
    simulator_config = SimulatorConfig(
        box_size=box_size,
        n_agents=n_agents,
        n_objects=n_objects,
        num_steps_lax=num_steps_lax,
        dt=dt,
        freq=freq,
        neighbor_radius=neighbor_radius,
        to_jit=to_jit,
        use_fori_loop=use_fori_loop
    )

    agent_configs = [
        AgentConfig(idx=i,
                    x_position=np.random.rand() * simulator_config.box_size,
                    y_position=np.random.rand() * simulator_config.box_size,
                    orientation=np.random.rand() * 2. * np.pi)
        for i in range(simulator_config.n_agents)
        ]

    object_configs = [
        ObjectConfig(idx=simulator_config.n_agents + i,
                    x_position=np.random.rand() * simulator_config.box_size,
                    y_position=np.random.rand() * simulator_config.box_size,
                    orientation=np.random.rand() * 2. * np.pi)
        for i in range(simulator_config.n_objects)
        ]

    state = converters.set_state_from_config_dict(
        {
            StateType.AGENT: agent_configs,
            StateType.OBJECT: object_configs,
            StateType.SIMULATOR: [simulator_config]
        }
        )

    saving_name = "test_dir"
    nve_states = []

    simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid)
    assert not simulator.recording

    lg.info("Running simulation")
    simulator.start_recording(saving_name)
    assert simulator.recording

    # Run the simulation for num_steps and save the nve_state
    for _ in range(num_steps):
        simulator.step()
        nve_states.append(simulator.state.nve_state)

    simulator.stop_recording()
    assert not simulator.recording
    assert not simulator.records

    loaded_nve_states = simulator.load(saving_name)
    assert loaded_nve_states

    lg.info("Simulation complete")

    # At the momet the saving and the loading works but there are numerical errors
    for state, loaded_state in zip(nve_states, loaded_nve_states):
        # This will therefore raise an error in the test :
        assert jnp.array_equal(state.position, loaded_state.position)
