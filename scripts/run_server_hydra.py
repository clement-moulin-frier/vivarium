import logging
import hydra
import numpy as np

from omegaconf import DictConfig, OmegaConf

import vivarium.simulator.behaviors as behaviors
from vivarium.controllers.config import SimulatorConfig, AgentConfig, ObjectConfig
from vivarium.simulator.sim_computation import StateType
from vivarium.simulator.simulator import Simulator
from vivarium.simulator.sim_computation import dynamics_rigid
from vivarium.controllers.converters import set_state_from_config_dict
from vivarium.simulator.grpc_server.simulator_server import serve

lg = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # logging.basicConfig(level=args.log_level.upper())
    # simulator_config = SimulatorConfig(
    #     box_size=args["box_size"],
    #     n_agents=args["n_agents"],
    #     n_objects=args["n_objects"],
    #     num_steps_lax=args["num_steps_lax"],
    #     dt=args["dt"],
    #     freq=args["freq"],
    #     neighbor_radius=args["neighbor_radius"],
    #     to_jit=args["to_jit"],
    #     use_fori_loop=args["use_fori_loop"]
    # )

    # agent_configs = [AgentConfig(idx=i,
    #                              x_position=np.random.rand() * simulator_config.box_size,
    #                              y_position=np.random.rand() * simulator_config.box_size,
    #                              orientation=np.random.rand() * 2. * np.pi)
    #                  for i in range(simulator_config.n_agents)]

    # object_configs = [ObjectConfig(idx=simulator_config.n_agents + i,
    #                                x_position=np.random.rand() * simulator_config.box_size,
    #                                y_position=np.random.rand() * simulator_config.box_size,
    #                                orientation=np.random.rand() * 2. * np.pi)
    #                   for i in range(simulator_config.n_objects)]

    # state = set_state_from_config_dict({StateType.AGENT: agent_configs,
    #                                     StateType.OBJECT: object_configs,
    #                                     StateType.SIMULATOR: [simulator_config]
    #                                     })

    # simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid)
    # serve(simulator)

if __name__ == '__main__':
    main(None)