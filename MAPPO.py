import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from env import MultiAgentDroneEnv  # Import your environment
from collections import deque
import numpy as np
import os

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_global_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

### CONFIGURATION ###
class Config:
    """
    Holds hyperparameters and settings for MAPPO training.
    Designed with real-world robotics constraints in mind.
    """
    def __init__(self):
        # RL hyperparameters
        self.num_episodes = 100              # Total number of episodes
        self.gamma = 0.99                    # Discount factor, balancing short-term and long-term rewards

        # NN training hyperparameters
        self.lr_actor = 1e-2                 # Learning rate for the actor network
        self.lr_critic = 1e-2                # Learning rate for the critic network
        self.entropy_coef = 0.1              # Entropy coefficient for exploration
        self.value_loss_coef = 0.5           # Coefficient for value loss
        self.max_grad_norm = 0.05            # Gradient clipping
        self.num_steps = 100                 # Steps per PPO update
        self.batch_size = 16                 # Batch size for PPO updates
        self.ppo_clip = 0.2                  # PPO clipping parameter
        self.epochs = 3                      # Epochs per PPO update for stability
        self.target_threshold = np.log(19)   # Target detection threshold, matches the environment

        # Multi-Agent Coordination
        self.centralized_critic = True       # Enable centralized critic for cooperative learning
        self.use_attention = False           # Optional: Use attention for state representation

        # Save and checkpoint settings
        self.checkpoint_dir = "checkpoints"  # Directory for saving models
        self.save_interval = 10              # Save model every 10 updates
        self.log_interval = 5                # Log training metrics every 5 updates

        # Real-World Constraints
        self.device = device                 # Device selection for training
        self.sim_to_real_transfer = True     # Enable configurations suitable for robotics applications
        self.observation_space_dim = None    # Placeholder for dynamically determined observation space
        self.action_space_dim = None         # Placeholder for dynamically determined action space

    def set_from_env(self, env):
        """
        Dynamically adjust settings based on the environment.
        :param env: The MultiAgentDroneEnv environment instance.
        """
        positions_dim = env.num_agents * 2  # x, y for each agent
        directions_dim = env.num_agents     # Direction for each agent
        map_dim = np.prod(env.observation_space.spaces['map'].shape)  # Flattened map dimensions
        likelihood_dim = np.prod(env.log_likelihood_ratios.shape)     # Log likelihood ratios
        self.observation_space_dim = positions_dim + directions_dim + map_dim + likelihood_dim
        self.action_space_dim = env.action_space.nvec.sum()

        # Ensure the checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)


'''
# Example initialization of the environment and configuration
if __name__ == "__main__":
    # Initialize the environment
    set_global_seed(42)
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
'''    

### ACTOR-CRITIC NETWORKS ###
class ActorNetwork(nn.Module):
    """
    The Actor network outputs a probability distribution over actions for each agent.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # Output probabilities for each action
        )
        
        # Add stabilization for the logits
        def forward(self, x):
            logits = self.network[:-1](x)  # Extract layers before softmax
            logits = torch.clamp(logits, min=-20, max=20)  # Stabilize logits
            return nn.functional.softmax(logits, dim=-1)

    def forward(self, x):
        return self.network(x)


class CriticNetwork(nn.Module):
    """
    The centralized Critic network evaluates the value of the global state.
    """
    def __init__(self, state_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output a single scalar value
        )

    def forward(self, x):
        return self.network(x)
    

'''
# Initialize networks based on the Config
obs_dim = config.observation_space_dim
action_dim = config.action_space_dim
state_dim = config.observation_space_dim  # Assuming the global state has the same dim as observation

actor = ActorNetwork(obs_dim, action_dim).to(config.device)
critic = CriticNetwork(state_dim).to(config.device)

print("Actor and Critic networks initialized:")
print(actor)
print(critic)
'''


class ReplayBuffer:
    """
    Stores experiences for PPO updates.
    """
    def __init__(self, num_agents, obs_dim, action_dim, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device

        # Buffers for storing experiences
        self.states = torch.zeros((buffer_size, num_agents, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, num_agents), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)
        self.values = torch.zeros((buffer_size, num_agents), dtype=torch.float32, device=device)

        # Current size tracker
        self.ptr = 0
        self.full = False

    def add(self, state, action, log_prob, reward, done, value):
        """
        Add an experience to the buffer.
        """
        idx = self.ptr % self.buffer_size
        self.states[idx] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[idx] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.log_probs[idx] = torch.tensor(log_prob, dtype=torch.float32, device=self.device)
        self.rewards[idx] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.dones[idx] = torch.tensor(done, dtype=torch.float32, device=self.device)
        self.values[idx] = torch.tensor(value, dtype=torch.float32, device=self.device)

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def sample(self):
        """
        Return the buffer's contents for PPO updates.
        """
        size = self.buffer_size if self.full else self.ptr
        return {
            "states": self.states[:size],
            "actions": self.actions[:size],
            "log_probs": self.log_probs[:size],
            "rewards": self.rewards[:size],
            "dones": self.dones[:size],
            "values": self.values[:size],
        }

    def clear(self):
        """
        Reset the buffer.
        """
        self.ptr = 0
        self.full = False
        

### TRAINING LOOP ###
def train(config, env, actor, critic, buffer):
    """
    The main training loop for MAPPO.
    :param config: Configuration object with hyperparameters.
    :param env: MultiAgentDroneEnv instance.
    :param actor: Actor network for policy updates.
    :param critic: Critic network for value updates.
    :param buffer: ReplayBuffer instance for storing experiences.
    """
    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=config.lr_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.lr_critic)
    

    # Initialize tracking metrics
    episode_rewards = []
    loss_critic_history = []
    loss_actor_history = []

    for episode in range(1, config.num_episodes + 1):
        print(f"Beginning episode #{episode}...")  # Start message
        # Reset environment
        state, _ = env.reset(seed=episode)
        done = False
        episode_reward = 0

        # Trajectory collection
        # Trajectory collection
        for t in range(config.num_steps):
            # Get actions and log probabilities from actor
            state_tensor = torch.tensor(state, dtype=torch.float32, device=config.device).view(1, -1)
            state_tensor = torch.clamp(state_tensor, min=-1e3, max=1e3)  # Clamp extreme values
            state_tensor = (state_tensor - state_tensor.mean()) / (state_tensor.std() + 1e-8)  # Normalize

            action_probs = actor(state_tensor).view(env.num_agents, -1)  # Reshape for multiple agents
            action_probs = torch.clamp(action_probs, min=1e-8)  # Avoid zero probabilities
            #print(f"Action probabilities shape: {action_probs.shape}")  # Debugging log
            
            actions = Categorical(action_probs).sample()  # Sample actions for each agent
            actions_np = actions.cpu().numpy()  # Convert to numpy for the environment
            log_probs = Categorical(action_probs).log_prob(actions)
            
            # Execute actions in the environment
            next_state, reward, done, _ = env.step(actions_np)  # Pass actions to env
            value = critic(state_tensor)
            
            # Log per timestep reward
            print(f"Reward at timestep {t + 1} = {np.sum(reward):.2f}")
            
            # Add to buffer
            buffer.add(
                state=state,
                action=actions_np,
                log_prob=log_probs.cpu().detach().numpy(),
                reward=reward,
                done=done,
                value=value.cpu().detach().numpy()
                )
            
            # Update state and rewards
            state = next_state
            episode_reward += np.sum(reward)
            
            if t == config.num_steps:
                print("Terminating episode because step {t+1} = {config.num_steps} is max step in PPO update")
                done = True
            
            if done:
                break


        # Process rewards and perform PPO update
        advantages, returns = calculate_advantages_and_returns(buffer, config.gamma)
        loss_actor, loss_critic = ppo_update(buffer, actor, critic, actor_optimizer, critic_optimizer, config, advantages, returns)

        # Log metrics
        episode_rewards.append(episode_reward)
        loss_actor_history.append(loss_actor)
        loss_critic_history.append(loss_critic)

        
        if episode % config.log_interval == 0:
            print(f"Ending FINAL Episode {episode}/{config.num_episodes}: Reward = {np.mean(episode_rewards[-config.log_interval:]):.2f}, "
                  f"Actor Loss = {np.mean(loss_actor_history[-config.log_interval:]):.4f}, "
                  f"Critic Loss = {np.mean(loss_critic_history[-config.log_interval:]):.4f}")
            
        else:
            # Print metrics after every episode
            print(f"Ending Episode {episode}/{config.num_episodes}: Average Reward = {np.mean(episode_rewards):.2f}, "
                  f"Actor Loss = {np.mean(loss_actor_history):.4f}, "
                  f"Critic Loss = {np.mean(loss_critic_history):.4f}")

        # Save model checkpoints
        if episode % config.save_interval == 0:
            torch.save(actor.state_dict(), os.path.join(config.checkpoint_dir, f"actor_{episode}.pth"))
            torch.save(critic.state_dict(), os.path.join(config.checkpoint_dir, f"critic_{episode}.pth"))

    return actor, critic


def calculate_advantages_and_returns(buffer, gamma):
    """
    Compute advantages and returns for PPO updates.
    :param buffer: ReplayBuffer instance containing trajectories.
    :param gamma: Discount factor for future rewards.
    """
    data = buffer.sample()
    rewards, dones, values = data['rewards'], data['dones'], data['values']
  
    values = torch.cat([data['values'], torch.zeros((1, data['values'].size(1)), device=buffer.device)], dim=0)


    advantages = torch.zeros_like(rewards, device=buffer.device)
    returns = torch.zeros_like(rewards, device=buffer.device)

    # Compute advantages and returns
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae * (1 - dones[t])
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    return advantages, returns


def ppo_update(buffer, actor, critic, actor_optimizer, critic_optimizer, config, advantages, returns):
    """
    Perform the PPO update step.
    :param buffer: ReplayBuffer instance containing trajectories.
    :param actor: Actor network.
    :param critic: Critic network.
    :param actor_optimizer: Optimizer for actor.
    :param critic_optimizer: Optimizer for critic.
    :param config: Configuration object.
    :param advantages: Computed advantages.
    :param returns: Computed returns.
    """
    data = buffer.sample()
    states, actions, old_log_probs = data['states'], data['actions'], data['log_probs']

    for _ in range(config.epochs):
        # Get current log probabilities and values
        action_probs = actor(states)
        new_log_probs = Categorical(action_probs).log_prob(actions)
        entropy = Categorical(action_probs).entropy()

        values = critic(states).squeeze(-1)

        # Calculate PPO objective
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - config.ppo_clip, 1 + config.ppo_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean() - config.entropy_coef * entropy.mean()

        value_loss = config.value_loss_coef * (returns - values).pow(2).mean()

        # Update actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
        actor_optimizer.step()

        # Update critic
        critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
        critic_optimizer.step()

    return actor_loss.item(), value_loss.item()
