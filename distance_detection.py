import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import math

# Constants for distance calculation
# These values would need to be calibrated for your specific camera
KNOWN_WIDTH = {
    'car': 1.8,  # average car width in meters
    'person': 0.5,  # average person width in meters
    'truck': 2.5,  # average truck width in meters
    'bus': 2.6,  # average bus width in meters
    'motorcycle': 0.8,  # average motorcycle width in meters
    'bicycle': 0.6,  # average bicycle width in meters
}
FOCAL_LENGTH = 800  # This is an assumed value, needs calibration

# Distance thresholds for warnings (in meters)
WARNING_THRESHOLD = 10  # Warning if distance is less than this
DANGER_THRESHOLD = 5   # Danger warning if distance is less than this

def calculate_distance(bbox_width, actual_width, focal_length=FOCAL_LENGTH):
    """
    Calculate distance using the triangle similarity principle
    
    Args:
        bbox_width (float): Width of the bounding box in pixels
        actual_width (float): Actual width of the object in meters
        focal_length (float): Focal length of the camera
        
    Returns:
        float: Estimated distance in meters
    """
    # Distance = (Actual Width * Focal Length) / Perceived Width
    if bbox_width == 0:
        return float('inf')  # Avoid division by zero
    
    distance = (actual_width * focal_length) / bbox_width
    return distance

def process_frame(frame, model):
    """
    Process a single frame to detect objects and calculate distances
    
    Args:
        frame (numpy.ndarray): Input image frame
        model (YOLO): YOLO model for object detection
        
    Returns:
        numpy.ndarray: Processed frame with detections and distance information
    """
    # Get original frame dimensions
    height, width = frame.shape[:2]
    
    # Run YOLOv10 inference
    results = model(frame)
    
    # Variables to track the closest object
    min_distance = float('inf')
    closest_warning_text = ""
    closest_warning_color = (0, 255, 0)  # Default green
    
    # Process detections
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Calculate box width
            bbox_width = x2 - x1
            
            # Get class and confidence
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            cls_name = model.names[cls_id]
            
            # Only process vehicles and people
            if cls_name in KNOWN_WIDTH:
                # Calculate distance
                distance = calculate_distance(bbox_width, KNOWN_WIDTH[cls_name])
                
                # Determine color based on distance
                if distance < DANGER_THRESHOLD:
                    color = (0, 0, 255)  # Red for danger
                    warning_text = f"DANGER! {distance:.1f}m"
                elif distance < WARNING_THRESHOLD:
                    color = (0, 165, 255)  # Orange for warning
                    warning_text = f"WARNING! {distance:.1f}m"
                else:
                    color = (0, 255, 0)  # Green for safe
                    warning_text = f"Safe: {distance:.1f}m"
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Add text with class name, confidence and distance
                label = f"{cls_name} {conf:.2f} - {distance:.1f}m"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Track the closest object that requires a warning
                if distance < WARNING_THRESHOLD and distance < min_distance:
                    min_distance = distance
                    closest_warning_text = warning_text
                    closest_warning_color = color
    
    # Add warning text at the top of the frame only for the closest object
    if min_distance < WARNING_THRESHOLD:
        cv2.putText(frame, closest_warning_text, (width // 2 - 100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, closest_warning_color, 2)
    
    return frame

def main():
    """Main function to run the distance detection system"""
    # Load YOLOv10 model
    print("Loading YOLOv10 model...")
    model = YOLO("yolov10n.pt")  # Use YOLOv10 nano model
    
    # Try to open webcam
    print("Opening video source...")
    try:
        cap = cv2.VideoCapture(0)  # Use webcam
        if not cap.isOpened():
            raise Exception("Could not open webcam")
    except Exception as e:
        print(f"Error opening webcam: {e}")
        print("Trying to use a sample image instead...")
        # Use a sample image from the dataset
        sample_img_path = "road_datas/images/training/1.jpg"
        frame = cv2.imread(sample_img_path)
        if frame is None:
            print(f"Could not read image from {sample_img_path}")
            return
        
        # Process the sample image
        processed_frame = process_frame(frame, model)
        
        # Save and display the result
        cv2.imwrite("result.jpg", processed_frame)
        print(f"Processed image saved as 'result.jpg'")
        
        # Try to display the image
        try:
            window_name = "Distance Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, processed_frame)
            print("Press any key or close the window to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("Could not display image. Result saved as 'result.jpg'")
        return
    
    print("Starting video processing. Press 'q' to quit or close the window.")
    
    # Create a named window that can be closed
    window_name = "Distance Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Process video frames
    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Process the frame
        processed_frame = process_frame(frame, model)
        
        # Display the result
        cv2.imshow(window_name, processed_frame)
        
        # Check if window is still open
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            running = False
        
        # Break the loop if 'q' is pressed or ESC key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            running = False
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()
