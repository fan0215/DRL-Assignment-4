import gymnasium # Or import gym if your dmc.py wrapper uses that
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Define the Actor Network Structure (must match the one used for training)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # A common MLP architecture for DDPG
        self.layer_1 = nn.Linear(state_dim, 400) # Or 256
        self.layer_2 = nn.Linear(400, 300) # Or 256
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        # Use tanh to output actions in [-1, 1], then scale by max_action
        x = torch.tanh(self.layer_3(x)) * self.max_action
        return x

# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts based on a learned DDPG policy for DMC Cartpole-Balance."""
    def __init__(self):
        # Environment properties for cartpole-balance (state observations)
        self.state_dim = 5    # As per problem description
        self.action_dim = 1   # As per problem description
        self.max_action = 1.0 # Action space is [-1.0, 1.0]

        # Initialize the Actor network
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action)

        # --- Load Pre-trained Weights ---
        # The evaluation script expects this file to exist.
        # Ensure this path is correct relative to where eval.py is run.
        self.model_path = "actor_dmc.pth" # Choose a distinct name for this model

        if os.path.exists(self.model_path):
            try:
                # Load weights to CPU for general compatibility.
                self.actor.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
                self.actor.eval()  # Set the network to evaluation mode
                print(f"Successfully loaded pre-trained DDPG actor model from {self.model_path}")
                self._model_loaded = True
            except Exception as e:
                print(f"Error loading DDPG actor model from {self.model_path}: {e}")
                print("Agent will act randomly.")
                self._model_loaded = False
        else:
            print(f"Warning: DDPG actor model file '{self.model_path}' not found.")
            print("Agent will act randomly.")
            self._model_loaded = False

        # Fallback random action space (matches the original stub)
        # Using gymnasium.spaces.Box as per the stub.
        # The action space from dmc.py might be np.float32, but this should be fine.
        if not self._model_loaded:
            self.action_space = gymnasium.spaces.Box(-self.max_action, self.max_action,
                                                     (self.action_dim,), np.float64)


    def act(self, observation):
        """
        Selects an action based on the observation using the loaded actor network.
        """
        if not self._model_loaded:
             # Fallback to random action if model loading failed or file not found
             return self.action_space.sample()

        # Convert observation (numpy array) to PyTorch tensor
        # Ensure dtype is float32 and add a batch dimension
        state = torch.FloatTensor(observation.astype(np.float32)).unsqueeze(0)

        # Get action from actor network
        with torch.no_grad(): # No gradient calculation needed during evaluation
            action_tensor = self.actor(state)

        # Convert action tensor back to numpy array
        # .cpu() moves tensor to CPU (if it was on GPU)
        # .detach() removes it from computation graph
        # .numpy() converts to numpy array
        # [0] selects the action from the batch dimension, result is shape (action_dim,)
        action_numpy = action_tensor.cpu().detach().numpy()[0]

        # The environment expects an action of shape (1,) for this specific task
        return action_numpy