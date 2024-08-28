import logging
import hydra

from omegaconf import DictConfig, OmegaConf

from vivarium.simulator.simulator import Simulator
from vivarium.environments.braitenberg.selective_sensing import SelectiveSensorsEnv, init_state
from vivarium.simulator.grpc_server.simulator_server import serve

lg = logging.getLogger(__name__)
update_freq = 100
num_steps_lax = 10

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig = None) -> None:
    logging.basicConfig(level=cfg.log_level)

    # TODO : Update args and use them here later
    args = OmegaConf.merge(cfg.default, cfg.scene)
    
    # init state and env
    state = init_state()
    env = SelectiveSensorsEnv(state=state)

    # init simulator
    simulator = Simulator(
        env_state=state, 
        env=env, 
        update_freq=update_freq, 
        num_steps_lax=num_steps_lax
    )

    lg.info('Simulator server started')
    serve(simulator)

if __name__ == '__main__':
    main()
