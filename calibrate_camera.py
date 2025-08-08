import cv2
import numpy as np
import argparse
import os

def calibrate_focal_length(image_path, known_distance, known_width, object_width_pixels):
    """
    Calibrate the focal length of the camera
    
    Args:
        image_path (str): Path to the calibration image
        known_distance (float): Known distance from camera to object in meters
        known_width (float): Known width of the object in meters
        object_width_pixels (int): Width of the object in pixels in the image
        
    Returns:
        float: Calculated focal length
    """
    # Focal Length = (Object Width in Pixels * Known Distance) / Known Width
    focal_length = (object_width_pixels * known_distance) / known_width
    return focal_length

def select_object(image_path):
    """
    Allow user to select an object in the image for calibration
    
    Args:
        image_path (str): Path to the calibration image
        
    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the selected region
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Create a window
    window_name = "Select Object for Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Initialize variables
    roi = None
    running = True
    
    # Define mouse callback function
    def draw_roi(event, x, y, flags, param):
        nonlocal roi, image
        
        # Clone the image to draw on
        img_copy = image.copy()
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing ROI
            roi = [(x, y)]
        
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            # Update ROI while dragging
            if roi:
                roi = [roi[0], (x, y)]
                # Draw rectangle
                cv2.rectangle(img_copy, roi[0], roi[1], (0, 255, 0), 2)
                cv2.imshow(window_name, img_copy)
        
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing ROI
            roi = [roi[0], (x, y)]
            # Draw final rectangle
            cv2.rectangle(img_copy, roi[0], roi[1], (0, 255, 0), 2)
            cv2.imshow(window_name, img_copy)
    
    # Set mouse callback
    cv2.setMouseCallback(window_name, draw_roi)
    
    # Display the image
    cv2.imshow(window_name, image)
    
    print("Select the object by clicking and dragging. Press 'c' to confirm, 'r' to reset, or ESC to cancel.")
    print("You can also close the window to cancel.")
    
    while running:
        key = cv2.waitKey(1) & 0xFF
        
        # Check if window is still open
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            roi = None
            running = False
        
        if key == ord('c') and roi:
            # Confirm selection
            running = False
        
        elif key == ord('r'):
            # Reset selection
            roi = None
            cv2.imshow(window_name, image)
        
        elif key == 27:  # ESC key
            # Cancel
            roi = None
            running = False
    
    cv2.destroyAllWindows()
    
    if roi:
        # Convert to (x1, y1, x2, y2) format
        x1, y1 = roi[0]
        x2, y2 = roi[1]
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Calibrate camera focal length for distance measurement')
    parser.add_argument('--image', type=str, default='', help='Path to calibration image')
    parser.add_argument('--distance', type=float, default=0, help='Known distance from camera to object in meters')
    parser.add_argument('--width', type=float, default=0, help='Known width of the object in meters')
    
    args = parser.parse_args()
    
    # If no image is provided, use a sample from the dataset
    if not args.image:
        sample_dir = "road_datas/images/training"
        if os.path.exists(sample_dir):
            sample_files = os.listdir(sample_dir)
            if sample_files:
                args.image = os.path.join(sample_dir, sample_files[0])
                print(f"Using sample image: {args.image}")
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found")
        return
    
    # If distance or width is not provided, ask for them
    if args.distance <= 0:
        try:
            args.distance = float(input("Enter the known distance from camera to object (in meters): "))
        except ValueError:
            print("Error: Invalid distance value")
            return
    
    if args.width <= 0:
        try:
            args.width = float(input("Enter the known width of the object (in meters): "))
        except ValueError:
            print("Error: Invalid width value")
            return
    
    # Select object in the image
    print("Please select the object in the image...")
    roi = select_object(args.image)
    
    if roi:
        x1, y1, x2, y2 = roi
        object_width_pixels = x2 - x1
        
        # Calculate focal length
        focal_length = calibrate_focal_length(args.image, args.distance, args.width, object_width_pixels)
        
        print(f"\nCalibration Results:")
        print(f"Object width in pixels: {object_width_pixels}")
        print(f"Calculated focal length: {focal_length:.2f}")
        print(f"\nUpdate the FOCAL_LENGTH constant in distance_detection.py with this value.")
        
        # Save calibration data
        with open("calibration_data.txt", "w") as f:
            f.write(f"Image: {args.image}\n")
            f.write(f"Known distance: {args.distance} meters\n")
            f.write(f"Known width: {args.width} meters\n")
            f.write(f"Object width in pixels: {object_width_pixels}\n")
            f.write(f"Calculated focal length: {focal_length:.2f}\n")
        
        print(f"Calibration data saved to 'calibration_data.txt'")
    else:
        print("Calibration cancelled or no object selected")

if __name__ == "__main__":
    main()
