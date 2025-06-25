"""
This module acts as an agent to interact with the Vintix model.
It handles model loading, context management, and action generation.
"""
import torch
import numpy as np
import logging
from collections import deque

# It's better to handle the python path in the execution script (run_vintix.py)
# but for standalone testing or simple cases, this can work.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vintix.vintix.vintix import Vintix

logger = logging.getLogger(__name__)

class VintixAgent:
    """
    A wrapper around the Vintix model to simplify interaction.
    """

    def __init__(self, model_path: str, context_length: int = 10, task_name: str = "ur5_pick_place_blue_cube"):
        """
        Initializes the Vintix agent.

        Args:
            model_path (str): Path to the directory containing the Vintix model files
                              (model.pth, config.json, metadata.json).
            context_length (int): The maximum length of the history (context) to maintain.
            task_name (str): The name of the task defined in the model's metadata.
        """
        self.model = Vintix()
        self.context_length = context_length
        self.task_name = task_name
        
        self.observations = deque(maxlen=self.context_length)
        self.actions = deque(maxlen=self.context_length)
        self.rewards = deque(maxlen=self.context_length)
        
        self.step_num = 0

        try:
            logger.info(f"Loading Vintix model from {model_path}...")
            self.model.load_model(model_path)
            self.model.eval()  # Set the model to evaluation mode
            logger.info("Vintix model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load Vintix model: {e}")
            raise

    def reset(self):
        """
        Resets the agent's context and step counter for a new episode.
        """
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.step_num = 0
        logger.info("Vintix agent context has been reset.")

    def get_action(self, current_observation: np.ndarray, prev_reward: float = 0.0) -> np.ndarray:
        """
        Gets the next action from the Vintix model based on the current observation.

        Args:
            current_observation (np.ndarray): The current state of the environment.
            prev_reward (float): The reward received after the previous action.

        Returns:
            np.ndarray: The action to be taken, as predicted by the model.
        """
        self.observations.append(torch.tensor(current_observation, dtype=torch.float32))
        self.rewards.append(torch.tensor([prev_reward], dtype=torch.float32))

        # We need a placeholder for the first action. A zero-action is a common choice.
        if not self.actions:
            action_dim = self.model.metadata[self.task_name]['acs_dim']
            self.actions.append(torch.zeros(action_dim, dtype=torch.float32))

        # Prepare model input
        model_input = [{
            "observation": torch.stack(list(self.observations)),
            "prev_action": torch.stack(list(self.actions)),
            "prev_reward": torch.stack(list(self.rewards)),
            "step_num": torch.arange(self.step_num, self.step_num + len(self.observations)),
            "task_name": self.task_name
        }]

        # Get action from the model
        predicted_action = self.model.get_action(model_input)[0]

        # Update context
        self.actions.append(torch.tensor(predicted_action, dtype=torch.float32))
        self.step_num += 1
        
        logger.info(f"Step {self.step_num}: Received action {predicted_action} for observation.")
        
        return predicted_action 