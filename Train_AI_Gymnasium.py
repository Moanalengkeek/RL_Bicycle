from stable_baselines3 import PPO
import gymnasium
import bicycle

def train_agent(run_length):
    env = gymnasium.make('bicycle/Bicycle-v0')
    # Create the PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        # clip_range=0.1,# 0.1,  # Adjust the clip range for exploration/exploitation balance
        # learning_rate=0.0001, # prev 0.1
        # gamma=0.9, # 0.9
        # n_steps=5,  #1000 Should be the maximum number of steps per episode
        # batch_size=5, #10
    )
    # Train the agent
    model.learn(total_timesteps=int(run_length), progress_bar=True)  # Adjust the number of timesteps as needed
    env. render()
    # Save the trained agent
    model.save("bicycle_rl_agent")


def retrain_agent(agent, run_length):
    env = gymnasium.make('bicycle/Bicycle-v0')
    model = PPO(
        "MlpPolicy",
        env,
        # clip_range=0.9,  # 0.1,  # Adjust the clip range for exploration/exploitation balance
        # ent_coef=0.01,  # 0.01,  # Increase entropy coefficient for more exploration
        # use_sde=True,  # Enable generalized State Dependent Exploration
        # sde_sample_freq=16,  # Adjust frequency of noise matrix sampling, Removing this makes many of the runs the same
        # learning_rate=0.01,  # prev 0.0003
        # gamma=0.99,
        # n_steps=2048,  # Should be the maxiumum number of steps per episode
        # # batch_size=64,
        # n_epochs=10,
        # policy_kwargs=policy_kwargs,
        # verbose=1  # Get it to print information
    )
    model.load(agent)
    model.learn(total_timesteps=int(run_length), progress_bar=True)
    env. render()
    model.save("bicycle_rl_agent_second")


if __name__ == '__main__':
    train_agent(1e5)


