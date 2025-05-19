import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os # Required for file path operations

# --- Constants and Device Setup (from train.py) ---
# These constants must match what was used during training
HIDDEN_SIZE = 512
STATE_SIZE = 67
ACTION_SIZE = 21

# Determine the device (CPU or CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Actor Network Definition (from train.py) ---
# This class defines the policy network that the agent will use to select actions.
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
        # Clamp log_std to ensure stable standard deviation values
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mean, std

    # This sample method is part of the Actor class but is primarily used during training.
    # For evaluation (which the 'act' method will perform), we'll use the deterministic mean.
    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        x_t = normal.rsample() # Reparameterization trick
        action = torch.tanh(x_t) # Apply tanh to bound actions between -1 and 1
        log_prob = normal.log_prob(x_t)
        # Adjust log_prob for the tanh squashing function
        log_prob -= torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# --- Agent Class for Submission ---
# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts based on a pre-trained SAC policy."""
    def __init__(self):
        # Define the action space, ensuring it matches the environment's expected action range
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACTION_SIZE,), np.float64)

        # Initialize the actor network and move it to the appropriate device
        self.actor = Actor(STATE_SIZE, ACTION_SIZE).to(device)

        # Load the pre-trained model weights
        self.load_model("best_SAC_ICM_model.pth")

        # Set the actor network to evaluation mode (disables dropout, batchnorm, etc.)
        self.actor.eval()

    def load_model(self, filename):
        """Loads the actor's state dictionary from the specified file."""
        # Define potential paths where the model might be located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.join(current_dir, "..")
        possible_paths = [
            os.path.join(current_dir, filename), # Current directory
            os.path.join(parent_dir, filename)    # Parent directory (common for training scripts)
        ]

        model_loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    # Load the checkpoint and the actor's state dictionary
                    checkpoint = torch.load(path, map_location=device)
                    self.actor.load_state_dict(checkpoint['actor_state_dict'])
                    print(f"Model successfully loaded from {path}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"Error loading model from {path}: {e}")

        if not model_loaded:
            raise FileNotFoundError(
                f"Model file '{filename}' not found in current directory or parent directory. "
                "Please ensure 'best_SAC_ICM_model.pth' is correctly placed."
            )

    def act(self, observation):
        """
        Selects an action based on the current observation using the pre-trained actor network.
        For evaluation, a deterministic action (mean of the policy distribution) is used.
        """
        # Convert the observation (numpy array) to a PyTorch tensor, add a batch dimension, and move to device
        state = torch.from_numpy(observation).float().to(device).unsqueeze(0)

        # Perform inference without tracking gradients
        with torch.no_grad():
            # Get the mean of the policy distribution (deterministic action)
            mean, _ = self.actor(state)
            # Apply tanh to squish the action to the range [-1, 1]
            action = torch.tanh(mean)

        # Convert the action tensor back to a numpy array and remove the batch dimension
        return action[0].cpu().numpy()