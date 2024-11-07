import time

from vivarium.controllers.notebook_controller import NotebookController
from vivarium.utils.handle_server_interface import start_server_and_interface, stop_server_and_interface

RUN_TIME = 7  
RESOURCE_SPAWN_PERIOD = 3  


# TODO : clean the test with pytest fixtures
def test_notebook_controller():
    start_server_and_interface(scene_name='session_3')

    def obstacle_avoidance(agent):
        """
        Simple obstacle avoidance behavior based on sensor data.
        """
        left, right = agent.sensors(sensed_entities=["obstacles"])
        return 1. - right, 1. - left

    def get_agents_positions():
        """
        Retrieves the positions of all agents in the simulation.
        """
        return [(agent.x_position, agent.y_position) for agent in controller.agents]

    def count_resources():
        """
        Counts the number of existing resources in the simulation.
        """
        resources_list = [entity for entity in controller.all_entities if entity.subtype_label == "resources" and entity.exists]
        return len(resources_list)

    # Initialize the controller
    controller = NotebookController()

    try:
        controller.run()

        # Initial states
        initial_positions = get_agents_positions()
        initial_resource_count = count_resources()
        print(f"Initial resource count: {initial_resource_count}")

        # Configure each agent
        for agent in controller.agents:
            agent.infos()
            agent.exists = True

            left, right = agent.sensors(sensed_entities=["robots", "obstacles"])
            print(f"Agent sensors - Left: {left}, Right: {right}")
            assert isinstance(left, (float, int))
            assert isinstance(right, (float, int))

            agent.detach_all_behaviors()
            agent.attach_behavior(obstacle_avoidance, interval=5, weight=2.0)
            agent.start_all_behaviors()

            agent.proxs_dist_max = 100.0
            agent.proxs_cos_min = 0.8

        controller.start_resources_apparition(period=RESOURCE_SPAWN_PERIOD, position_range=((0, 50), (100, 200)))

        # Run simulation for the specified time
        time.sleep(RUN_TIME)

        # Final states
        final_positions = get_agents_positions()
        final_resource_count = count_resources()
        spawned_resources = final_resource_count - initial_resource_count
        expected_spawned_resources = (RUN_TIME // RESOURCE_SPAWN_PERIOD) + 1 # add 1 because the first resource is spawned at the start

        assert initial_positions != final_positions, "Agents should have moved from their initial positions."
        assert spawned_resources == expected_spawned_resources, "Unexpected number of spawned resources."

    finally:
        controller.stop()
        stop_server_and_interface()
