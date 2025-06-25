"""
Test script for the Vision module.
"""
import cv2
import logging
import sys
import os

# Add the project root to the Python path to allow importing 'vision'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision import Vision

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Initializes the camera using the Vision module, detects colored objects,
    and displays the results in real-time.
    """
    # Use camera_id=0 for the default system camera. Change if you have multiple cameras.
    vision_system = Vision(camera_id=0)

    if not vision_system.initialize_camera():
        logging.critical("Could not initialize the vision system. Is the camera connected?")
        return

    logging.info("Camera initialized. Press 'q' in the display window to quit.")

    # Visualization colors for different objects
    display_colors = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0)
    }

    while True:
        # 1. Get a frame from the camera
        frame = vision_system.get_frame()
        if frame is None:
            logging.warning("Failed to get a frame. Retrying...")
            continue

        # 2. Find objects using the vision module
        detected_objects = vision_system.find_colored_objects(frame)

        # 3. Log detections to the console
        # Log only if there are any detected objects to avoid spamming the console
        if any(detected_objects.values()):
            logging.info(f"Detected: {detected_objects}")

        # 4. Visualize the results
        display_frame = frame.copy()
        for color, objects in detected_objects.items():
            draw_color = display_colors.get(color, (255, 255, 255))  # Default to white
            for (x, y) in objects:
                cv2.circle(display_frame, (x, y), 10, draw_color, 2)
                cv2.putText(display_frame, color, (x + 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)
        
        cv2.imshow("Vision Module Test", display_frame)

        # 5. Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Quit command received.")
            break

    # Clean up
    vision_system.release_camera()
    cv2.destroyAllWindows()
    logging.info("Test finished successfully.")

if __name__ == "__main__":
    main() 