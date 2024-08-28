import logging
import hydra

from omegaconf import DictConfig, OmegaConf

from vivarium.environments.braitenberg.selective_sensing import init_state, SelectiveSensorsEnv
from vivarium.simulator.simulator import Simulator

lg = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig = None) -> None:
    logging.basicConfig(level=cfg.log_level)

    # TODO : Update args and update them later
    args = OmegaConf.merge(cfg.default, cfg.scene)
    
    state = init_state()
    env = SelectiveSensorsEnv(state=state)
    simulator = Simulator(env_state=state, env=env)

    lg.info("Running simulation")
    simulator.run(threaded=False, num_steps=cfg.num_steps)
    lg.info("Simulation complete")

if __name__ == "__main__":
    main()

