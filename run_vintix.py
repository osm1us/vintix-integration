"""
Main executable file for the Vintix-powered robot control loop.
Orchestrates all modules to run the robot.
"""
import logging
import queue
import time
import cv2
import numpy as np

# --- Module Imports ---
from robot_controller import RobotController
from vintix_agent import VintixAgent
from vision import Vision
from voice_control import VoiceCommandHandler
from coordinate_mapper import CoordinateMapper
from datalogger import HDF5Logger

# --- Configuration ---
LOG_LEVEL = logging.INFO
ESP32_IP = "192.168.1.10"  # !!! IMPORTANT: Change this to your ESP32's IP address !!!
URDF_PATH = "manipulator.urdf"
VINTIX_MODEL_PATH = "models/vintix_checkpoint"
SUCCESS_THRESHOLD_METERS = 0.02  # 2 сантиметра до цели
MAX_EPISODE_STEPS = 200 # Максимальное количество шагов в одной попытке

# Basic logging setup
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Corobot")


class CorobotSystem:
    """The main class that orchestrates the entire robot system."""

    def __init__(self):
        logger.info("Initializing Corobot System...")
        self.running = True
        self.target_color = None
        self.is_episode_active = False
        self.episode_step_counter = 0
        self.command_queue = queue.Queue()

        # Initialize core components
        try:
            self.robot = RobotController(urdf_path=URDF_PATH, esp32_ip=ESP32_IP)
            self.agent = VintixAgent(model_path=VINTIX_MODEL_PATH)
            self.vision = Vision(camera_id=0)
            self.voice = VoiceCommandHandler(command_queue=self.command_queue)
            self.mapper = CoordinateMapper()
            self.logger = HDF5Logger(log_dir="data/training_logs")

            if not self.vision.initialize_camera():
                raise ConnectionError("Failed to initialize camera.")

        except Exception as e:
            logger.critical(f"Failed to initialize a core component: {e}")
            self.running = False

    def run(self):
        """Starts the main control loop."""
        if not self.running:
            logger.critical("System cannot run due to initialization failure.")
            return

        self.voice.start_listening()
        logger.info("System is running. Press 'q' in the OpenCV window to quit.")

        try:
            while self.running:
                # 1. Get a fresh frame from the camera
                frame = self.vision.get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                # 2. Process inputs (voice commands)
                self._handle_voice_commands()

                # 3. Perceive the environment (Vision)
                all_objects = self.vision.find_colored_objects(frame)
                
                # 4. Decide and Act (Vintix + Robot Control)
                if self.is_episode_active:
                    if self.target_color and self.target_color in all_objects and all_objects[self.target_color]:
                        # Цель видна, выполняем шаг Vintix
                        self._vintix_step(all_objects)
                    else:
                        # Цель пропала во время выполнения
                        logger.warning(f"Target '{self.target_color}' lost. Ending episode as failure.")
                        self._end_episode(success=False)
                
                # 5. Visualize
                self._visualize(frame, all_objects)

                # 6. User exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
        finally:
            self.shutdown()

    def _vintix_step(self, all_objects):
        """Performs one step of the Vintix control loop."""
        # a. Get target coordinates
        target_pixel_coords = all_objects[self.target_color][0] # Take the first detected object
        
        # b. Map to robot's world coordinates (currently using a placeholder mapping)
        # NOTE: This mapping needs to be calibrated for real-world accuracy.
        target_world_coords = self.mapper.pixel_to_robot(target_pixel_coords[0], target_pixel_coords[1])
        
        # c. Get robot's current state
        current_joint_angles = self.robot.get_current_angles_rad()

        # d. Form the observation vector for the agent
        # The observation space for Vintix is often a concatenation of robot state and goal state
        observation = np.concatenate([
            np.array(current_joint_angles),
            np.array(target_world_coords)
        ]).astype(np.float32)
        
        # e. Get action from Vintix
        # We are not implementing a reward function yet, so reward is always 0.
        action_rad = self.agent.get_action(observation, prev_reward=0.0)
        
        # g. Log the step data
        self.logger.log_step(observation, action_rad, 0.0, self.episode_step_counter)
        self.episode_step_counter += 1
        
        # h. Execute the action
        self.robot.move_to_angles_rad(action_rad.tolist())

        # i. Check for success condition
        end_effector_pos = self.robot.get_end_effector_position()
        distance_to_target = np.linalg.norm(np.array(end_effector_pos) - np.array(target_world_coords))
        logger.debug(f"Step: {self.episode_step_counter}, Distance to target: {distance_to_target:.4f}m")

        if distance_to_target < SUCCESS_THRESHOLD_METERS:
            logger.info(f"Success! Reached target within {SUCCESS_THRESHOLD_METERS}m threshold.")
            self._end_episode(success=True)
        
        elif self.episode_step_counter >= MAX_EPISODE_STEPS:
            logger.warning(f"Max steps ({MAX_EPISODE_STEPS}) reached. Ending episode as failure.")
            self._end_episode(success=False)

    def _handle_voice_commands(self):
        """Check the queue for voice commands and update state."""
        try:
            command = self.command_queue.get_nowait()
            logger.info(f"Received command: {command}")
            action = command.get('action')

            if action == 'grab':
                target = command.get('target')
                if target in ['red', 'green', 'blue']:
                    logger.info(f"Starting new episode for target: {target}")
                    self.target_color = target
                    self._start_new_episode()
            
            elif action == 'stop':
                logger.info("User requested STOP. Ending episode as failure.")
                if self.is_episode_active:
                    self._end_episode(success=False)
                self.robot.go_home()

            elif action == 'home':
                if self.is_episode_active:
                    logger.warning("Returning home, current episode is cancelled (failure).")
                    self._end_episode(success=False)
                self.robot.go_home()

        except queue.Empty:
            # No commands, which is normal
            pass

    def _start_new_episode(self):
        """Resets necessary states and components for a new task."""
        self.is_episode_active = True
        self.episode_step_counter = 0
        self.agent.reset()  # Reset agent's context (short-term memory)
        self.logger.reset_episode_buffer() # Reset our logger
        logger.info("Episode started. Logger and Agent are ready.")

    def _end_episode(self, success: bool):
        """Handles the end of an episode, logging and resetting state."""
        if not self.is_episode_active:
            return # Avoid multiple endings

        final_reward = 1.0 if success else -1.0
        logger.info(f"Ending episode. Success: {success}, Final Reward: {final_reward}")
        
        self.logger.finish_episode(final_reward)
        self.is_episode_active = False
        self.target_color = None

        if success:
            # Perform the actual grab and return home
            logger.info("Performing grab action and returning home.")
            self.robot.set_gripper(open_gripper=False) # Close gripper
            time.sleep(1)
            self.robot.go_home()
            time.sleep(1)
            self.robot.set_gripper(open_gripper=True) # Open gripper at home

    def _visualize(self, frame, all_objects):
        """Displays the camera feed and overlays debug information."""
        # Draw all detected objects
        for color, objects in all_objects.items():
            for (x, y) in objects:
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

        # Highlight the current target
        if self.target_color and self.target_color in all_objects and all_objects[self.target_color]:
            x, y = all_objects[self.target_color][0]
            cv2.circle(frame, (x, y), 15, (0, 255, 0), 2)
            cv2.putText(frame, f"TARGET: {self.target_color}", (x + 20, y + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Corobot Vintix Control", frame)


    def shutdown(self):
        """Properly closes all resources."""
        logger.info("Shutting down Corobot System...")
        if self.is_episode_active:
            # Ensure any active episode is logged as a failure on shutdown
            logger.warning("System shutting down during an active episode. Logging as failure.")
            self._end_episode(success=False)
            
        self.voice.stop_listening()
        self.vision.release_camera()
        self.robot.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = CorobotSystem()
    system.run()
    logger.info("System has shut down.") 