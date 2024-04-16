import logging
import hydra

import hydra.core
import hydra.core.global_hydra
from omegaconf import DictConfig, OmegaConf

from vivarium.simulator import behaviors
from vivarium.simulator.states import init_state_from_dict
from vivarium.simulator.simulator import Simulator
from vivarium.simulator.physics_engine import dynamics_rigid
from vivarium.simulator.grpc_server.simulator_server import serve

lg = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig = None) -> None:
    logging.basicConfig(level=cfg.log_level)

    args = OmegaConf.merge(cfg.default, cfg.scene)

    state = init_state_from_dict(args)

    simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid)

    # necessary to be able to load other scenes
    glob = hydra.core.global_hydra.GlobalHydra()
    glob.clear()

    lg.info('Simulator server started')
    serve(simulator)

if __name__ == '__main__':
    main()
