import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from tqdm import tqdm # For progress bar

# --- Define Networks --- (Must match student_agent.py)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        x = torch.tanh(self.layer_3(x)) * self.max_action
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Critic takes state and action as input
        self.layer_1 = nn.Linear(state_dim + action_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 1) # Outputs a single Q-value

    def forward(self, state, action):
        # Concatenate state and action
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
        # Convert to tensors
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
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005, buffer_size=1000000):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1e-2) # Weight decay can help stabilize critic

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).cpu().data.numpy().flatten()
        noise = np.random.normal(0, self.max_action * noise_scale, size=action.shape)
        action = (action + noise).clip(-self.max_action, self.max_action)
        return action


    def train(self, batch_size=100):
        if len(self.replay_buffer) < batch_size:
            return # Don't train if buffer doesn't have enough samples

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        # --- Critic Update ---
        # Get target Q-value
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss (MSE)
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        # Compute actor loss (negative Q-value from critic)
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update Target Networks (Soft Updates) ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


    def save_actor(self, filename="actor.pth"):
        torch.save(self.actor.state_dict(), filename)
        print(f"Actor model saved to {filename}")

# --- Training Loop ---
if __name__ == "__main__":
    env_name = "Pendulum-v1"
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Hyperparameters (tune these!)
    num_episodes = 200       # Number of training episodes
    max_steps_per_episode = 200 # Max steps per episode for Pendulum-v1
    lr_actor = 3e-4         # Actor learning rate
    lr_critic = 3e-4        # Critic learning rate
    gamma = 0.99            # Discount factor
    tau = 0.005             # Target network soft update rate
    buffer_size = 100000    # Replay buffer capacity
    batch_size = 64         # Training batch size
    start_timesteps = 1000  # Number of steps with random actions before training starts
    exploration_noise = 0.1 # Std deviation of Gaussian exploration noise

    agent = DDPGAgent(state_dim, action_dim, max_action, lr_actor, lr_critic, gamma, tau, buffer_size)

    total_timesteps = 0
    episode_rewards = []

    print(f"Starting training on {env_name}...")
    print(f"State Dim: {state_dim}, Action Dim: {action_dim}, Max Action: {max_action}")

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False # Pendulum specific

        for step in range(max_steps_per_episode):
            total_timesteps += 1

            # Select action: random for initial steps, otherwise policy + noise
            if total_timesteps < start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise_scale=exploration_noise)

            # Perform action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated # Combine termination conditions

            # Store experience in replay buffer
            # Use truncated flag to indicate if episode ended early (important for value estimation)
            # We store 'done' as True if terminated OR truncated for Q-learning target calculation
            real_done_for_buffer = float(terminated) # Only truly done if terminated, not truncated
            agent.replay_buffer.add(state, action, reward, next_state, real_done_for_buffer)

            state = next_state
            episode_reward += reward

            # Train agent after collecting start_timesteps
            if total_timesteps >= start_timesteps:
                agent.train(batch_size)

            if done:
                break

        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:]) # Moving average of last 100 episodes
        print(f"Episode: {episode+1}, Timesteps: {step+1}, Reward: {episode_reward:.2f}, Avg Reward (last 100): {avg_reward:.2f}")

    # Save the trained actor model
    agent.save_actor("actor.pth")
    env.close()
    print("Training finished.")