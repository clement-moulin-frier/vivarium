import logging
import hydra

from omegaconf import DictConfig, OmegaConf

from vivarium.simulator import behaviors
from vivarium.simulator.states import init_simulator_state
from vivarium.simulator.states import init_agent_state
from vivarium.simulator.states import init_object_state
from vivarium.simulator.states import init_entities_state
from vivarium.simulator.states import init_state
from vivarium.simulator.simulator import Simulator
from vivarium.simulator.physics_engine import dynamics_rigid
from vivarium.simulator.grpc_server.simulator_server import serve

lg = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig = None) -> None:
    logging.basicConfig(level=cfg.log_level)

    args = OmegaConf.merge(cfg.default, cfg.scene)

    simulator_state = init_simulator_state(**args.simulator)

    agents_state = init_agent_state(simulator_state=simulator_state, **args.agents)

    objects_state = init_object_state(simulator_state=simulator_state, **args.objects)

    entities_state = init_entities_state(simulator_state=simulator_state, **args.entities)

    state = init_state(
        simulator_state=simulator_state,
        agents_state=agents_state,
        objects_state=objects_state,
        entities_state=entities_state
        )

    simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid)

    lg.info('Simulator server started')
    serve(simulator)

if __name__ == '__main__':
    main()
