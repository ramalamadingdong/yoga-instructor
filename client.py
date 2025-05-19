import requests
import tensorflow as tf
import numpy as np
import cv2
import sys

POSITION_URL = "http://localhost:5000/get_position"
INSTRUCTION_URL = "http://localhost:5000/get_instructions"

try:
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path='hrnet_pose-hrnetpose-w8a8.tflite')
    delegate = tf.lite.load_delegate('libQnnTFLiteDelegate.so', { 'backend_type': 'htp' })
    interpreter.modify_graph_with_delegate(delegate)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)


def pose_estimation(image):
    """Get pose keypoints from the frame using TFLite HRNet"""
    # Preprocess the image
    input_size = (256, 256)  # HRNet input size
    img = cv2.resize(image, input_size)
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
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
        #Get target yoga position from server
        target_pose = get_position()
        print(f"Waiting for human to assume {target_pose}...")
        
        # Get current pose estimation
        current_pose = pose_estimation()
        
        # Compare poses
        position = pose_classification(current_pose)
        
        if position == target_pose:
            get_pose_hold_instructions(target_pose)
            print("Get infromation from Deepseek server about said position then SAY it") 

        else:
            print("Say not quite there yet")
        
if __name__ == "__main__":
    main()

