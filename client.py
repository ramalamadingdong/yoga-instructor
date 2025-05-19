import requests


POSITION_URL = "http://localhost:5000/get_position"
INSTRUCTION_URL = "http://localhost:5000/get_instructions"

def pose_estimation():
    """Get pose keypoints from the frame using TFLite HRNet"""
    #TODO: Implement pose estimation
    keypoints = []
    return keypoints

def pose_classification(current_pose):
    """Check if the current pose matches the target yoga position"""
    #TODO: Implement yoga position check
    return True

def get_position():
    response = requests.get(POSITION_URL)
    position = response.json()
    return position

def get_pose_hold_instrutions():
    response = requests.get(INSTRUCTION_URL)
    position = response.json()
    return position

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
            get_pose_hold_instrutions(target_pose)
            print("Get infromation from Deepseek server about said position then SAY it") 

        else:
            print("Say not quite there yet")
        
if __name__ == "__main__":
    main()

