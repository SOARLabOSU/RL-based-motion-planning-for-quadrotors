import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from env import MultiAgentDroneEnv  # Import your environment
from collections import deque
import numpy as np
import os

from MAPPO import (
    Config, 
    set_global_seed, 
    MultiAgentDroneEnv, 
    ActorNetwork, 
    CriticNetwork, 
    ReplayBuffer, 
    train
)

# Set the seed for reproducibility
SEED = 42
set_global_seed(SEED)

# Initialize the environment
env = MultiAgentDroneEnv(
    agent_positions=[(1, 99, 120), (99, 1, 270)],
    target_positions=[(40, 90), (70, 35)],
    grid_size=(100, 100),
    radius_field_of_view=5,
    target_threshold=np.log(19),
    max_steps=100,
    likelihood_decay=0.1
)

# Initialize the configuration
config = Config()
config.set_from_env(env)

print("Configuration initialized:")
print(f"Observation space dimension: {config.observation_space_dim}")
print(f"Action space dimension: {config.action_space_dim}")
print(f"Device: {config.device}")

# Initialize the actor and critic networks
actor = ActorNetwork(config.observation_space_dim, config.action_space_dim).to(config.device)
critic = CriticNetwork(config.observation_space_dim).to(config.device)

print("Actor and Critic networks initialized:")
print(actor)
print(critic)

# Initialize the replay buffer
buffer_size = config.num_steps  # Size matches the number of steps per update
buffer = ReplayBuffer(
    num_agents=env.num_agents,
    obs_dim=config.observation_space_dim,
    action_dim=config.action_space_dim,
    buffer_size=buffer_size,
    device=config.device
)

# Run the training loop
if __name__ == "__main__":
    print("Starting training...")
    trained_actor, trained_critic = train(config, env, actor, critic, buffer)
    print("Training complete!")

    # Save the final models
    torch.save(trained_actor.state_dict(), "final_actor.pth")
    torch.save(trained_critic.state_dict(), "final_critic.pth")
    print("Models saved!")
