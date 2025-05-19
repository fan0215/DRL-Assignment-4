import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os # Import os to check for file existence

# Define the Actor Network Structure (must match the one used for training)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # Simple MLP architecture
        self.layer_1 = nn.Linear(state_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, action_dim)
        self.max_action = max_action # Store max action value

    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        # Use tanh to output actions in [-1, 1], then scale
        x = torch.tanh(self.layer_3(x)) * self.max_action
        return x

# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts based on a learned DDPG policy."""
    def __init__(self):
        # Environment properties (Pendulum-v1 specific)
        # These should ideally be fetched from the env, but hardcoding is okay
        # for this specific problem if the env is fixed.
        self.state_dim = 3  # Observation space dim for Pendulum-v1
        self.action_dim = 1 # Action space dim for Pendulum-v1
        self.max_action = 2.0 # Max action value for Pendulum-v1

        # Initialize the Actor network
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action)

        # --- Load Pre-trained Weights ---
        # The evaluation script expects this file to exist in the same directory
        self.model_path = "actor.pth"

        if os.path.exists(self.model_path):
            try:
                # Load the weights. Ensure loading to CPU for broader compatibility,
                # unless you know the evaluation environment has a specific device.
                self.actor.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
                self.actor.eval()  # Set the network to evaluation mode (important!)
                print(f"Successfully loaded pre-trained model from {self.model_path}")
                self._model_loaded = True
            except Exception as e:
                print(f"Error loading model from {self.model_path}: {e}")
                print("Agent will act randomly.")
                self._model_loaded = False
        else:
            print(f"Warning: Model file '{self.model_path}' not found.")
            print("Agent will act randomly.")
            # Fallback: Use random actions if model not found
            self.action_space = gym.spaces.Box(-self.max_action, self.max_action, (self.action_dim,), np.float32)
            self._model_loaded = False


    def act(self, observation):
        """
        Selects an action based on the observation using the loaded actor network.
        """
        if not self._model_loaded:
             # Fallback to random action if model loading failed or file not found
             return self.action_space.sample()

        # Convert observation (numpy array) to PyTorch tensor
        # Ensure dtype is float32 and add a batch dimension (unsqueeze(0))
        state = torch.FloatTensor(observation.reshape(1, -1)) # Reshape just in case

        # Get action from actor network
        # No gradient calculation needed during evaluation
        with torch.no_grad():
            action = self.actor(state)

        # Convert action (tensor) back to numpy array
        # .cpu() moves tensor to CPU (if it was on GPU)
        # .detach() removes it from computation graph
        # .numpy() converts to numpy array
        # [0] selects the action from the batch dimension
        # Ensure the output format matches what env.step expects (often a numpy array or list)
        # For Pendulum-v1, an array like [action_value] is expected.
        return action.cpu().detach().numpy()[0]