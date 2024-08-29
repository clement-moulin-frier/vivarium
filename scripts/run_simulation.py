import logging
import hydra

from omegaconf import DictConfig, OmegaConf

from vivarium.environments.braitenberg.selective_sensing import init_state, SelectiveSensorsEnv
from vivarium.simulator.simulator import Simulator

lg = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig = None) -> None:
    logging.basicConfig(level=cfg.log_level)

    # init state and env
    args = OmegaConf.merge(cfg.default, cfg.scene)
    state = init_state(**args)
    env = SelectiveSensorsEnv(state=state)

    # init simulator
    simulator = Simulator(env_state=state, env=env)

    # run it
    lg.info("Running simulation")
    simulator.run(threaded=False, num_steps=cfg.num_steps)
    lg.info("Simulation complete")

if __name__ == "__main__":
    main()

