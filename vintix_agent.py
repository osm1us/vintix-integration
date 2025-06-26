"""
This module acts as an agent to interact with the Vintix model.
It handles model loading, context management, and action generation.
"""
import torch
import numpy as np
import logging
from vintix.vintix import Vintix
from config import load_config, settings
from utils import get_logger

logger = logging.getLogger(__name__)

class VintixAgent:
    """
    A wrapper for the Vintix model to facilitate inference and action generation.
    Handles model loading, state reset, and provides a clean interface 
    for getting actions based on observations.
    """

    def __init__(self, model_path: str):
        """
        Initializes the VintixAgent and loads the Vintix model.

        Args:
            model_path (str): Path to the Vintix model checkpoint directory.
        """
        self.logger = get_logger(__name__)
        self.logger.info("Initializing VintixAgent...")

        try:
            self.vintix = Vintix(
                model_path=model_path,
                task_name='industrial-benchmark-0-v1',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            self.logger.info(f"Vintix model loaded successfully for task 'industrial-benchmark-0-v1' on device '{self.vintix.device}'.")
            self.reset()
        except Exception as e:
            self.logger.critical(f"Failed to load Vintix model from '{model_path}': {e}", exc_info=True)
            raise

    def reset(self):
        """
        Resets the agent's internal state, including the context for the Vintix model.
        This should be called at the beginning of each new episode.
        """
        self.vintix.reset()
        self.logger.info("VintixAgent state has been reset.")

    def get_next_action(self, joint_angles: np.ndarray, last_reward: float = 0.0) -> np.ndarray:
        """
        Gets the next action from the Vintix model based on the current observation.

        Args:
            joint_angles (np.ndarray): The current joint angles (6,) of the robot in radians.
            last_reward (float): The reward from the previous step.

        Returns:
            np.ndarray: The calculated action delta (3,) for the first three joint angles.
        """
        if not isinstance(joint_angles, np.ndarray):
            raise TypeError(f"joint_angles must be a numpy array, but got {type(joint_angles)}")

        if joint_angles.shape != (6,):
            raise ValueError(f"Expected joint_angles to have shape (6,), but got {joint_angles.shape}")
            
        # The model for this task expects only the 6 joint angles as observation.
        observation = joint_angles

        try:
            action = self.vintix.get_next_action(observation, last_reward)
            self.logger.debug(f"Vintix Observation (shape {observation.shape}) -> Action (shape {action.shape}): {action}")
            return action
        except Exception as e:
            self.logger.error(f"Error getting next action from Vintix model: {e}", exc_info=True)
            # Return a zero action in case of an error to prevent crashing
            return np.zeros(3)