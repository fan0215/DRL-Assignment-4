import sys
import os
# Assuming dmc.py is in the parent directory of the current script's directory
# Adjust if your directory structure is different
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dmc import make_dmc_env
except ImportError:
    print("Error: Could not import make_dmc_env from dmc.py.")
    print("Please ensure dmc.py is in the correct path (e.g., parent directory).")
    sys.exit(1)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from tqdm import tqdm

# --- Define Networks --- (Must match student_agent.py Actor)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        x = torch.tanh(self.layer_3(x)) * self.max_action
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.layer_1(sa))
        q = F.relu(self.layer_2(q))
        q = self.layer_3(q)
        return q

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

# --- DDPG Agent for Training ---
class DDPGAgentTrainer:
    def __init__(self, state_dim, action_dim, max_action, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005, buffer_size=1000000):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1e-2) # From OpenAI Baselines

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim # Store action_dim for noise generation

    def select_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)) # Ensure state is float32
        action = self.actor(state).cpu().data.numpy().flatten()
        # Add exploration noise
        noise = noise_scale * self.max_action * np.random.randn(self.action_dim)
        action = (action + noise).clip(-self.max_action, self.max_action)
        return action

    def train(self, batch_size=100):
        if len(self.replay_buffer) < batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        # Critic Update
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + ((1 - done) * self.gamma * target_Q)

        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Update
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Target Network Soft Updates
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_actor(self, filename="actor_dmc.pth"):
        torch.save(self.actor.state_dict(), filename)
        print(f"Actor model saved to {filename}")

# --- Training Loop ---
if __name__ == "__main__":
    env_name = "cartpole-balance"
    # Seed for make_dmc_env is for the environment's internal randomness during setup
    # For episodic reproducibility during training, reset with a seed if supported.
    env = make_dmc_env(env_name, seed=np.random.randint(0, 1000000), flatten=True, use_pixels=False)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # dmc.py wrapper usually sets action_space.high/low correctly
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0]) # Should be -max_action

    print(f"Obs shape: {env.observation_space.shape}, Action shape: {env.action_space.shape}")
    print(f"Max action: {max_action}, Min action: {min_action}")


    # Hyperparameters (tune these!)
    num_episodes = 500        # More episodes might be needed
    # Max steps per episode for cartpole-balance is typically 1000
    # The wrapper should handle time limits, env.step returns truncated=True
    lr_actor = 1e-4
    lr_critic = 1e-3
    gamma = 0.99
    tau = 0.005
    buffer_size = int(1e6)
    batch_size = 256          # Larger batch sizes often work well for DDPG
    start_timesteps = 10000   # Number of steps for random actions before training
    exploration_noise = 0.1   # Std deviation of Gaussian exploration noise

    # For seeding action selection and other random processes in the script
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


    agent = DDPGAgentTrainer(state_dim, action_dim, max_action, lr_actor, lr_critic, gamma, tau, buffer_size)

    total_timesteps = 0
    episode_rewards_deque = deque(maxlen=100) # For tracking average reward

    print(f"Starting DDPG training on DMC {env_name} (state observations)...")

    for episode in range(num_episodes):
        # The seed in make_dmc_env is for initial setup.
        # For reproducible episodes during training, env.reset() would need a seed argument.
        # If the dmc.py wrapper doesn't support env.reset(seed=...),
        # then each episode starts based on the environment's internal state.
        try:
            # Try to seed the reset if the wrapper supports it
            state, info = env.reset(seed=episode)
        except TypeError:
            # Fallback if env.reset doesn't take a seed
            state, info = env.reset()

        episode_reward = 0
        # DMC environments typically run for 1000 steps if not solved/failed earlier
        # The wrapper should handle this via truncated flag
        max_episode_steps = 1000 # Typical for cartpole-balance

        for step in range(max_episode_steps):
            total_timesteps += 1

            if total_timesteps < start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(np.array(state), noise_scale=exploration_noise)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done_for_buffer = terminated # For DDPG, 'done' means terminal state, not just time limit

            agent.replay_buffer.add(state, action, reward, next_state, float(done_for_buffer))

            state = next_state
            episode_reward += reward

            if total_timesteps >= start_timesteps:
                agent.train(batch_size)

            if terminated or truncated: # Episode ends if terminated or truncated
                break

        episode_rewards_deque.append(episode_reward)
        avg_reward = np.mean(episode_rewards_deque)
        print(f"Episode: {episode+1}, Total T: {total_timesteps}, Reward: {episode_reward:.2f}, Avg Reward (last 100): {avg_reward:.2f}")

        # Optionally save model periodically or based on performance
        if (episode + 1) % 50 == 0 or avg_reward > 900: # Example condition
            agent.save_actor("actor_dmc.pth")


    agent.save_actor("actor_dmc.pth") # Save final model
    env.close()
    print("Training finished.")