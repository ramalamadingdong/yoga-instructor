import subprocess, logging, time
import tensorflow as tf
import numpy as np
import cv2
from yapper import Yapper

# Use FastMCP again
from mcp.server import FastMCP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

yapper = Yapper()

try:
  delegate = tf.lite.experimental.load_delegate('libQnnTFLiteDelegate.so')
except:
    delegate = None
    logger.warning("Error loading LibQnnTFLiteDelegate.so")

if delegate:
  interpreter = tf.lite.Interpreter(
      model_path='hrnet_pose-hrnetpose-w8a8.tflite',
      experimental_delegates=[delegate])
else:
  interpreter = tf.lite.Interpreter(model_path='hrnet_pose-hrnetpose-w8a8.tflite')

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print model input requirements
logger.info("Model input details:", input_details[0])
logger.info("Model output details:", output_details[0])

# Allocate tensors
interpreter.allocate_tensors()

# Initialize FastMCP server
# The name is used for identification in clients
mcp = FastMCP("Rubik MCP Server (FastMCP)")

@mcp.tool()
async def talk_to_yogi(text_to_say: str) -> str:
    """Announces the next yoga position to be performed using text-to-speech.
    
    Args:
        text_to_say (str): The name of the yoga position to announce
        
    Returns:
        str: Confirmation message with the announced position
    """
    yapper.yap(text_to_say, plain=True)
    return "Announced position: " + text_to_say


@mcp.tool()
async def yoga_pose_estimation() -> str:
    """Performs yoga pose estimation using HRNet model to detect keypoints in an image.
    
    Returns:
        numpy.ndarray: Array of detected keypoint coordinates [x, y] for the pose
    """
    image = cv2.imread("test.jpg")

    if image is None:
        print("Error: Could not read image file")
        return None
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


@mcp.tool()
async def get_rubik_sw_info() -> dict:
    """This tool provides information about the Rubik board software capabilities (Linux kernel version and Jetpack version)."""
    logger.info("Executing get_rubik_sw_info tool...")
    
    sw_info = {
        "jetpack_release": "N/A",
        "linux_version": "N/A",
        "errors": []
    }

    commands = {
        "jetpack_release": "cat /etc/nv_tegra_release",
        "linux_version": "cat /proc/version"
    }

    for key, command in commands.items():
        logger.info(f"Executing command: {command}")
        try:
            result = subprocess.run(
                command,
                shell=True, # Use shell=True carefully
                check=True,
                capture_output=True,
                text=True
            )
            sw_info[key] = result.stdout.strip()
            logger.info(f"Command '{command}' successful.")
        except FileNotFoundError:
            error_msg = f"File not found for command: {command}"
            logger.error(error_msg)
            sw_info["errors"].append(error_msg)
            sw_info[key] = "Error: File not found"
        except subprocess.CalledProcessError as e:
            error_msg = f"Error executing command '{command}': {e.stderr.strip() or e}"
            logger.error(error_msg)
            sw_info["errors"].append(error_msg)
            sw_info[key] = f"Error: {e.stderr.strip() or e}"
        except Exception as e:
            error_msg = f"An unexpected error occurred while running command '{command}': {e}"
            logger.exception(error_msg)
            sw_info["errors"].append(error_msg)
            sw_info[key] = f"Error: Unexpected error"

    # Remove errors key if empty
    if not sw_info["errors"]:
        del sw_info["errors"]

    return sw_info

# --- Resource Implementations (using decorators) ---

@mcp.resource("rubik://info")
async def get_rubik_info() -> dict:
    """Provides basic information about the Rubik MCP server."""
    logger.info("Executing get_rubik_info resource...")
    # Use the name string directly instead of relying on mcp.title
    server_name = "Rubik MCP Server (FastMCP)"
    # Hardcode the list of capabilities since mcp.get_tools() doesn't exist
    capabilities_list = ["get_rubik_hw_info", "get_rubik_sw_info", "yoga_pose_estimation", "talk_to_yogi"]
    return {
        "server_name": server_name,
        "version": "0.3.0",
        "description": "MCP Server for a Yoga coach to interact with a Rubik board (using FastMCP/SSE).",
        "capabilities": capabilities_list
    }

if __name__ == "__main__":
    logger.info("Starting Rubik MCP Server via mcp.run() in SSE mode...")
    try:
        # Pass transport, host, and port directly to the run method
        mcp.run(transport='sse', host='0.0.0.0', port=8000, log_level='info') # Added log_level
    except TypeError as e:
        logger.error(f"Failed to start server with host/port/log_level arguments: {e}")
        logger.info("Attempting to start server with transport='sse' only (might bind to localhost:8000)...")
        try:
            # Fallback if host/port aren't accepted directly
             mcp.run(transport='sse')
        except Exception as e2:
            logger.exception(f"Failed to start server even with fallback: {e2}")
