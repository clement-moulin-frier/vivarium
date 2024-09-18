from vivarium.environments.braitenberg.selective_sensing import init_state, SelectiveSensorsEnv

NUM_STEPS = 10

def test_env_occlusion():
    """ Test the initialization of state without arguments """
    state = init_state()
    env = SelectiveSensorsEnv(state=state, occlusion=True)

    for _ in range(NUM_STEPS):
        state = env.step(state=state)

    assert env
    assert state

def test_env_no_occlusion():
    """ Test the initialization of state without arguments """
    state = init_state()
    env = SelectiveSensorsEnv(state=state, occlusion=False)

    for _ in range(NUM_STEPS):
        state = env.step(state=state)

    assert env
    assert state

