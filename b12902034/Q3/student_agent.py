import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os # Required for file path operations

# --- Constants and Device Setup ---
# These constants must exactly match what was used during training
HIDDEN_SIZE = 512
STATE_SIZE = 67  # Corresponds to env.observation_space.shape[0] for 'humanoid-walk'
ACTION_SIZE = 21 # Corresponds to env.action_space.shape[0] for 'humanoid-walk'

# Determine the device (CPU or CUDA) for running the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Actor Network Definition ---
# This class defines the neural network architecture for the policy (Actor).
# It's copied directly from your train.py to ensure consistency.
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(HIDDEN_SIZE, action_size)
        self.log_std_layer = nn.Linear(HIDDEN_SIZE, action_size)

    def forward(self, state):
        x = self.backbone(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # Clamp log_std to ensure numerical stability for standard deviation
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mean, std

    # The 'sample' method is used during training for exploration.
    # For inference in a deployed agent, we typically use the deterministic mean.
    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        x_t = normal.rsample() # Reparameterization trick for gradients
        action = torch.tanh(x_t) # Apply tanh to bound actions between -1 and 1
        log_prob = normal.log_prob(x_t)
        # Adjust log_prob for the tanh squashing function
        log_prob -= torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# --- Agent Class for Submission ---
# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """
    Agent that acts based on a pre-trained Soft Actor-Critic (SAC) policy.
    It loads the Actor network from a saved checkpoint.
    """
    def __init__(self):
        # Define the action space to match the environment's expected action range
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACTION_SIZE,), np.float64)

        # Initialize the Actor network and move it to the configured device (CPU/CUDA)
        self.actor = Actor(STATE_SIZE, ACTION_SIZE).to(device)

        # Load the pre-trained model weights for the Actor.
        # This filename should match the one used in agent.save() in train.py.
        self.load_model("best_SAC_ICM_model.pth")

        # Set the actor network to evaluation mode. This disables dropout, batch normalization,
        # and other layers that behave differently during training vs. inference.
        self.actor.eval()

    def load_model(self, filename):
        """
        Loads the actor's state dictionary from the specified file.
        It checks common locations (current directory, parent directory)
        and handles different saving key formats for robustness.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.join(current_dir, "..")
        possible_paths = [
            os.path.join(current_dir, filename), # Try current directory first
            os.path.join(parent_dir, filename)    # Then try parent directory
        ]

        model_loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    # Load the entire checkpoint dictionary
                    checkpoint = torch.load(path, map_location=device)

                    # Attempt to load using the new key ('actor') from your latest train.py save format
                    if 'actor' in checkpoint:
                        self.actor.load_state_dict(checkpoint['actor'])
                        print(f"Model successfully loaded from {path} using key 'actor'.")
                        model_loaded = True
                        break
                    # Fallback to the old key ('actor_state_dict') from previous train.py save format
                    elif 'actor_state_dict' in checkpoint:
                        self.actor.load_state_dict(checkpoint['actor_state_dict'])
                        print(f"Model successfully loaded from {path} using key 'actor_state_dict'.")
                        model_loaded = True
                        break
                    else:
                        # If the file exists but neither key is found, it's a warning
                        print(f"Warning: Model file {path} found, but neither 'actor' nor 'actor_state_dict' key found.")
                except Exception as e:
                    # Catch any other loading errors (e.g., corrupted file)
                    print(f"Error loading model from {path}: {e}")

        if not model_loaded:
            # If after checking all paths and keys, the model isn't loaded, raise an error.
            raise FileNotFoundError(
                f"Model file '{filename}' not found at expected paths "
                "(current directory or parent directory), or the required 'actor'/'actor_state_dict' key was missing."
            )

    def act(self, observation):
        """
        Selects an action based on the current observation using the pre-trained actor network.
        For evaluation, a deterministic action (the mean of the policy distribution) is used.
        """
        # Convert the observation (numpy array) to a PyTorch tensor.
        # Add an unsqueeze(0) to create a batch dimension (e.g., (67,) becomes (1, 67)).
        # Move the tensor to the correct device (CPU/CUDA).
        state = torch.from_numpy(observation).float().to(device).unsqueeze(0)

        # Perform inference without tracking gradients for efficiency
        with torch.no_grad():
            # Get the mean of the policy distribution from the actor network
            mean, _ = self.actor(state)
            # Apply tanh to squish the action to the valid range of [-1, 1]
            action = torch.tanh(mean)

        # Convert the action tensor back to a numpy array and remove the batch dimension
        # (e.g., (1, 21) becomes (21,)).
        return action[0].cpu().numpy()