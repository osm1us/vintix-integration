"""
Module for low-level motor control.
Handles the communication with the ESP32 bridge to send commands
to the Arduino Nano controlling the stepper motors and servo.
"""
import requests
import json
import logging
import time

logger = logging.getLogger(__name__)


class MotorController:
    """
    Manages communication with the ESP32 controller, sending raw commands
    for stepper motors and the gripper servo.
    """

    def __init__(self, ip_address: str, port: int = 80, timeout: int = 5, retries: int = 3):
        """
        Initializes the motor controller.

        Args:
            ip_address (str): The IP address of the ESP32.
            port (int): The port for the web server on the ESP32.
            timeout (int): Request timeout in seconds.
            retries (int): Number of retries for a failed request.
        """
        self.base_url = f"http://{ip_address}:{port}"
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        logger.info(f"MotorController initialized for ESP32 at {self.base_url}")

    def send_commands(self, command_list: list[str]) -> bool:
        """
        Sends a list of raw string commands to the ESP32.

        The ESP32 expects a JSON payload in the format: {"commands": ["CMD1", "CMD2", ...]}

        Args:
            command_list (list[str]): A list of commands to be executed.

        Returns:
            bool: True if the commands were sent successfully, False otherwise.
        """
        if not command_list:
            logger.warning("Command list is empty. Nothing to send.")
            return True

        url = f"{self.base_url}/command"
        payload = {"commands": command_list}
        
        for attempt in range(self.retries):
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
                
                logger.info(f"Successfully sent commands: {command_list}. Response: {response.text}")
                return True
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to send commands on attempt {attempt + 1}/{self.retries}. Error: {e}")
                time.sleep(1)  # Wait for 1 second before retrying

        logger.critical(f"Could not send commands to {url} after {self.retries} attempts.")
        return False

    def move_steppers(self, p1: int, p2: int, p3: int, p4: int) -> bool:
        """
        Creates and sends a command to move the stepper motors to target positions.

        Args:
            p1, p2, p3, p4 (int): The absolute step positions for the four stepper motors.

        Returns:
            bool: True if the command was sent successfully, False otherwise.
        """
        command = f"A P{p1} P{p2} P{p3} P{p4}"
        return self.send_commands([command])

    def set_gripper_angle(self, angle: int) -> bool:
        """
        Creates and sends a command to set the gripper servo angle.

        Args:
            angle (int): The target angle for the gripper servo (0-180).

        Returns:
            bool: True if the command was sent successfully, False otherwise.
        """
        # Clamping the angle to the valid range for the servo
        clamped_angle = max(0, min(angle, 180))
        if clamped_angle != angle:
            logger.warning(f"Gripper angle {angle} was clamped to {clamped_angle}.")

        command = f"G P{clamped_angle}"
        return self.send_commands([command])

    def check_connection(self) -> bool:
        """
        Checks the connection to the ESP32 by sending a request to its root path.

        Returns:
            bool: True if the ESP32 is reachable, False otherwise.
        """
        try:
            response = self.session.get(self.base_url, timeout=self.timeout)
            response.raise_for_status()
            logger.info(f"Successfully connected to ESP32 at {self.base_url}. Response: {response.text}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection check failed for {self.base_url}. Error: {e}")
            return False

    def close_session(self):
        """Closes the requests session."""
        self.session.close()
        logger.info("MotorController session closed.") 