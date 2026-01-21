"""Configuration settings for the simulation."""

# Simulation parameters
SIMULATION_TIME_STEP = 0.01  # 10ms time steps
USE_GUI = False  # Enable visualization

# Conveyor belt parameters
CONVEYOR_LENGTH = 2.0  # meters
CONVEYOR_WIDTH = 0.5   # meters
CONVEYOR_HEIGHT = 0.8  # meters
CONVEYOR_SPEED = 0.05  # m/s
PICKUP_POSITION = 1.0  # Position along belt for pickup

# Cable parameters
CONNECTOR_LENGTH = 0.05     # 5cm
CONNECTOR_WIDTH = 0.02      # 2cm
CONNECTOR_HEIGHT = 0.015    # 1.5cm
CABLE_LENGTH = 0.3          # 30cm
CABLE_DIAMETER = 0.005      # 5mm
CABLE_FLEXIBILITY = 0.3     # 0=rigid, 1=very flexible

# Camera parameters
CAMERA_POSITION = [1.0, 0.0, 2.0]  # Position above belt
CAMERA_LOOK_AT = [1.0, 0.0, 0.8]   # Look at belt surface
CAMERA_FOV = 60.0                  # Field of view in degrees
IMAGE_WIDTH = 640                  # Image width in pixels
IMAGE_HEIGHT = 480                 # Image height in pixels

# Robot parameters
ROBOT_HOME_POSITION = [1.0, 0.0, 1.5]  # Home position above belt
GRIPPER_WIDTH = 0.08                    # 8cm gripper opening
MAX_REACH = 0.8                         # Maximum reach distance
ROBOT_SPEED = 0.2                       # Movement speed m/s

# Vision parameters
DEPTH_THRESHOLD = 0.05      # 5cm above belt for detection
MIN_CABLE_LENGTH = 0.03     # Minimum 3cm cable length
MIN_CABLE_WIDTH = 0.01      # Minimum 1cm cable width

# Training parameters
NUM_TRAINING_EPISODES = 100
CABLES_PER_EPISODE = 5
MAX_STEPS_PER_EPISODE = 10000
