"""
This module is responsible for all computer vision tasks,
including camera handling and object detection.
"""
import cv2
import numpy as np
import logging
from data.color_ranges import COLOR_RANGES

logger = logging.getLogger(__name__)


class Vision:
    """
    Handles camera input and performs color-based object detection.
    """

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """
        Initializes the Vision system.

        Args:
            camera_id (int): The ID of the camera to use.
            width (int): The desired width of the camera frame.
            height (int): The desired height of the camera frame.
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.camera = None
        
        # Kernel for morphological operations
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def initialize_camera(self) -> bool:
        """
        Opens the camera device and sets its parameters.

        Returns:
            bool: True if the camera was initialized successfully, False otherwise.
        """
        logger.info(f"Initializing camera with ID {self.camera_id}...")
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            logger.critical("Failed to open camera.")
            return False

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce latency

        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"Camera initialized. Resolution: {actual_width}x{actual_height}")
        
        if actual_width != self.width or actual_height != self.height:
             logger.warning("Camera does not support the requested resolution. Using default.")
             
        return True

    def get_frame(self) -> np.ndarray | None:
        """
        Captures a single frame from the camera.

        Returns:
            np.ndarray | None: The captured frame as a NumPy array, or None on failure.
        """
        if not self.camera or not self.camera.isOpened():
            logger.error("Camera is not initialized.")
            return None

        # Grab and retrieve to get the most recent frame
        self.camera.grab()
        ret, frame = self.camera.read()

        if not ret or frame is None:
            logger.error("Failed to grab frame from the camera.")
            return None

        return frame

    def find_colored_objects(self, frame: np.ndarray, min_area: int = 500) -> dict[str, list[tuple[int, int]]]:
        """
        Finds objects of predefined colors in the given frame.

        Args:
            frame (np.ndarray): The input frame (in BGR format).
            min_area (int): The minimum contour area to be considered an object.

        Returns:
            A dictionary where keys are color names and values are lists of
            (x, y) coordinates for the center of each detected object.
            Example: {'red': [(100, 150)], 'blue': []}
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_objects = {color: [] for color in COLOR_RANGES}

        for color_name, ranges in COLOR_RANGES.items():
            mask = cv2.inRange(hsv_frame, ranges["lower"], ranges["upper"])
            
            # For red, we might have two ranges. This is a placeholder if we need it.
            # if color_name == 'red' and 'lower2' in ranges:
            #     mask2 = cv2.inRange(hsv_frame, ranges["lower2"], ranges["upper2"])
            #     mask = cv2.bitwise_or(mask, mask2)

            # Clean up the mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        detected_objects[color_name].append((cx, cy))
        
        return detected_objects

    def release_camera(self):
        """Releases the camera device."""
        if self.camera and self.camera.isOpened():
            self.camera.release()
            logger.info("Camera released.") 