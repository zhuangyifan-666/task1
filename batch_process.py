import cv2
import os
import argparse
import time
from ultralytics import YOLO
from distance_detection import process_frame

def process_directory(input_dir, output_dir, model_path="yolov10n.pt", limit=None):
    """
    Process all images in a directory
    
    Args:
        input_dir (str): Directory containing input images
        output_dir (str): Directory to save processed images
        model_path (str): Path to YOLO model
        limit (int, optional): Maximum number of images to process
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLO model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(input_dir, file))
    
    # Limit the number of images if specified
    if limit is not None and limit > 0:
        image_files = image_files[:limit]
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    start_time = time.time()
    for i, image_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            continue
        
        # Process image
        processed_image = process_frame(image, model)
        
        # Save processed image
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"processed_{filename}")
        cv2.imwrite(output_path, processed_image)
        
        print(f"Saved processed image to {output_path}")
    
    # Calculate and print processing time
    total_time = time.time() - start_time
    avg_time = total_time / len(image_files) if image_files else 0
    print(f"\nProcessing complete!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Process a batch of images for distance detection')
    parser.add_argument('--input', type=str, default='road_datas/images/training', 
                        help='Input directory containing images')
    parser.add_argument('--output', type=str, default='output', 
                        help='Output directory for processed images')
    parser.add_argument('--model', type=str, default='yolov10n.pt', 
                        help='Path to YOLO model')
    parser.add_argument('--limit', type=int, default=None, 
                        help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    process_directory(args.input, args.output, args.model, args.limit)

if __name__ == "__main__":
    main()
