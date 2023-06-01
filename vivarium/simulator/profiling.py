import cProfile
import pstats
from pstats import SortKey


from vivarium.simulator import config
from vivarium.simulator.simulator import EngineConfig, Simulator
from vivarium.simulator.sim_computation import dynamics_rigid


simulation_config = config.SimulatorConfig()

engine_config = EngineConfig(simulation_config=simulation_config, dynamics_fn=dynamics_rigid)

simulator = Simulator(engine_config=engine_config)


def run(s):
    for i in range(10):
        s.agent_configs[0].param.update(wheel_diameter=i / 100.)
        s.engine_config.update_agent_configs_from_state()
        s._run(num_loops=1)


cProfile.run('run(simulator)', 'profiler_stats')

p = pstats.Stats('profiler_stats')
p.sort_stats(SortKey.CUMULATIVE).print_stats()
