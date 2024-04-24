import logging
import hydra

from omegaconf import DictConfig, OmegaConf

from vivarium.simulator import behaviors
from vivarium.simulator.states import init_state
from vivarium.simulator.simulator import Simulator
from vivarium.simulator.physics_engine import dynamics_rigid

lg = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig = None) -> None:
    logging.basicConfig(level=cfg.log_level)

    args = OmegaConf.merge(cfg.default, cfg.scene)
    state = init_state(args)
    simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid)

    lg.info("Running simulation")
    simulator.run(threaded=False, num_steps=cfg.num_steps)
    lg.info("Simulation complete")

if __name__ == "__main__":
    main()
