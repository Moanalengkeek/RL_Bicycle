from gymnasium import register

register(
    id='bicycle/Bicycle-v0',
    entry_point='bicycle.env.bicycle_env:BicycleEnv',
    max_episode_steps=100,
)