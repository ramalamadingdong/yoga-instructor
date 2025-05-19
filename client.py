import requests
import tensorflow as tf
import numpy as np
import cv2
import sys

POSITION_URL = "http://localhost:5000/get_position"
INSTRUCTION_URL = "http://localhost:5000/get_instructions"

try:
  delegate = tf.lite.experimental.load_delegate('libQnnTFLiteDelegate.so')
except:
    delegate = None
    print("Error loading delegate")

if delegate:
  interpreter = tf.lite.Interpreter(
      model_path='hrnet_pose-hrnetpose-w8a8.tflite',
      experimental_delegates=[delegate])
else:
  interpreter = tf.lite.Interpreter(model_path='hrnet_pose-hrnetpose-w8a8.tflite')

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Allocate tensors
interpreter.allocate_tensors()

# Print model input requirements
print("Model input details:", input_details[0])
print("Model output details:", output_details[0])

print("Model loaded")
def pose_estimation(image):
    """Get pose keypoints from the frame using TFLite HRNet"""
    # Preprocess the image
    input_size = (192, 256)  # HRNet input size (width, height)
    img = cv2.resize(image, input_size)
    # Keep as uint8 since model expects uint8 input
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)
    
    # Run inference
    interpreter.invoke()

    # Get output tensor
    heatmaps = interpreter.get_tensor(output_details[0]['index'])[0]
    
    keypoints = []
    # Convert heatmaps to keypoint coordinates
    for heatmap in heatmaps:
        # Find the location of maximum value in heatmap
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        # Scale coordinates back to original image size
        x = x * image.shape[1] / input_size[0]
        y = y * image.shape[0] / input_size[1]
        keypoints.append([x, y])
    
    return np.array(keypoints)

def pose_classification(current_pose):
    """Check if the current pose matches the target yoga position"""
    #TODO: Implement yoga position check
    return True

def get_position():
    try:
        response = requests.get(POSITION_URL)
        response.raise_for_status()
        position = response.json()
        return position
    except requests.RequestException as e:
        print(f"Error getting position: {e}")
        return None

def get_pose_hold_instructions():
    try:
        response = requests.get(INSTRUCTION_URL)
        response.raise_for_status()
        position = response.json()
        return position
    except requests.RequestException as e:
        print(f"Error getting instructions: {e}")
        return None

def main():
    # Initialize video
    while True:
        try:
            #Get target yoga position from server
            target_pose = get_position()
            if target_pose is None:
                print("Waiting for server connection...")
                continue
                
            print(f"Waiting for human to assume {target_pose}...")
            
            # Read and process the image
            image = cv2.imread("test.jpg")
            if image is None:
                print("Error: Could not read image file")
                continue
                
            # Get current pose estimation
            current_pose = pose_estimation(image)
            
            # Compare poses
            position = pose_classification(current_pose)
            
            if position == target_pose:
                instructions = get_pose_hold_instructions()
                if instructions:
                    print(instructions)
            else:
                print("Not quite there yet")
                
        except Exception as e:
            print(f"Error occurred: {e}")
            continue

if __name__ == "__main__":
    main()

