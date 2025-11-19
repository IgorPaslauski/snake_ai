from snake_ai.env.snake_env import SnakeEnv, SnakeConfig


def test_env_reset_and_step():
    cfg = SnakeConfig(grid_width=10, grid_height=10)
    env = SnakeEnv(cfg)
    state = env.reset()
    assert state.shape[0] == 11

    next_state, reward, done, info = env.step(1)
    assert next_state.shape[0] == 11
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "score" in info
