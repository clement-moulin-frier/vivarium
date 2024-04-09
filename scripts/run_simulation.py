import hydra
import logging

from omegaconf import DictConfig, OmegaConf

from vivarium.simulator import behaviors
from vivarium.simulator.states import init_simulator_state
from vivarium.simulator.states import init_agent_state
from vivarium.simulator.states import init_object_state
from vivarium.simulator.states import init_nve_state
from vivarium.simulator.states import init_state
from vivarium.simulator.simulator import Simulator
from vivarium.simulator.sim_computation import dynamics_rigid

lg = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig = None) -> None:
    logging.basicConfig(level=cfg.log_level)

    args = OmegaConf.merge(cfg.default, cfg.scene)

    simulator_state = init_simulator_state(**args.simulator)

    agents_state = init_agent_state(simulator_state=simulator_state, **args.agents)

    objects_state = init_object_state(simulator_state=simulator_state, **args.objects)

    nve_state = init_nve_state(simulator_state=simulator_state, **args.nve)

    state = init_state(simulator_state=simulator_state,
                       agents_state=agents_state,
                       objects_state=objects_state,
                       nve_state=nve_state)

    simulator = Simulator(state, behaviors.behavior_bank, dynamics_rigid)

    lg.info("Running simulation")

    simulator.run(threaded=False, num_steps=cfg.num_steps)

    lg.info("Simulation complete")

if __name__ == "__main__":
    main()
