"""
High-level controller for the Corobot.

This module orchestrates the kinematics and motor control modules to provide
a simple interface for controlling the robot arm.
"""

import logging
from kinematics import Kinematics
from motor_control import MotorController

logger = logging.getLogger(__name__)

class RobotController:
    """
    Orchestrates robot movements by translating high-level commands
    into low-level motor signals.
    """

    def __init__(self, urdf_path: str, esp32_ip: str):
        """
        Initializes the Robot Controller.

        Args:
            urdf_path (str): Path to the URDF file for the robot model.
            esp32_ip (str): IP address of the ESP32 controller.
        """
        logger.info("Initializing RobotController...")
        try:
            self.kinematics = Kinematics(urdf_path)
            logger.info("Kinematics module initialized.")
        except Exception as e:
            logger.critical(f"Failed to initialize Kinematics: {e}")
            raise

        self.motor_controller = MotorController(ip_address=esp32_ip)
        if not self.motor_controller.check_connection():
            logger.warning("Could not establish connection with the motor controller on initialization.")
        else:
            logger.info("MotorController initialized and connection verified.")

        # Store the number of active (stepper) joints
        self.num_stepper_joints = len(self.kinematics.get_active_joint_indices())
        self.current_angles_rad = [0.0] * self.num_stepper_joints
        self.go_home() # Start at a known position

    def get_current_angles_rad(self) -> list[float]:
        """
        Returns the last commanded joint angles.
        NOTE: This is the last commanded state, not a reading from an encoder.
        """
        return self.current_angles_rad

    def move_to_angles_rad(self, joint_angles_rad: list[float]) -> bool:
        """
        Moves the robot's stepper-driven joints to the specified angles in radians.

        Args:
            joint_angles_rad (list[float]): A list of target angles for the active joints.
                                            The list length must match the number of active joints.

        Returns:
            bool: True if the movement command was sent successfully, False otherwise.
        """
        if len(joint_angles_rad) != self.num_stepper_joints:
            logger.error(f"Incorrect number of angles provided. Expected {self.num_stepper_joints}, got {len(joint_angles_rad)}.")
            return False

        try:
            steps = self.kinematics.radians_to_steps(joint_angles_rad)
            logger.info(f"Converted radians {joint_angles_rad} to steps {steps}.")

            if len(steps) != 4:
                 logger.error(f"Kinematics returned an unexpected number of steps. Expected 4, got {len(steps)}.")
                 return False

            success = self.motor_controller.move_steppers(p1=steps[0], p2=steps[1], p3=steps[2], p4=steps[3])
            if success:
                logger.info("Successfully sent move command.")
                # Update the current state only on successful command send
                self.current_angles_rad = joint_angles_rad
            else:
                logger.error("Failed to send move command.")
            return success

        except Exception as e:
            logger.error(f"An error occurred while trying to move to angles: {e}")
            return False

    def set_gripper(self, angle: int) -> bool:
        """
        Sets the gripper to a specific angle.

        Args:
            angle (int): The target angle (0-180). 0 is closed, 180 is open.

        Returns:
            bool: True if the command was sent successfully.
        """
        logger.info(f"Setting gripper to angle {angle}.")
        return self.motor_controller.set_gripper_angle(angle)

    def open_gripper(self) -> bool:
        """Opens the gripper fully."""
        return self.set_gripper(180)

    def close_gripper(self) -> bool:
        """Closes the gripper fully."""
        return self.set_gripper(0)
    
    def go_home(self) -> bool:
        """
        Moves the robot to its home position (all joint angles at 0 radians).
        """
        logger.info("Sending robot to home position.")
        home_angles_rad = [0.0] * self.num_stepper_joints
        return self.move_to_angles_rad(home_angles_rad)

    def shutdown(self):
        """Safely shuts down the controller connections."""
        logger.info("Shutting down RobotController.")
        self.motor_controller.close_session() 