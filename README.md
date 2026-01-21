# Luganodynamics - Robotic Cable Pickup Simulation

A Python simulation environment for training robotic hands to pick up cables from conveyor belts using a **Simulation-First approach**.

## Overview

This project simulates a robotic hand picking up cables with connectors from a moving conveyor belt. The system uses RGB-D (color + depth) images from an overhead camera to detect cables and estimate their pose for reliable grasping.

### Key Features

- **Flexible Cable Simulation**: Cables are modeled as elongated, partly flexible objects that can be curved or twisted
- **RGB-D Vision Processing**: Simulates overhead camera with depth sensing for cable detection and pose estimation
- **Robotic Hand Control**: Gripper control with inverse kinematics and grasp planning
- **Training Framework**: Collect data and metrics for training machine learning models
- **Simulation-First Approach**: Train in simulation before deploying to real hardware

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/danutaparaficz2/Luganodynamics.git
cd Luganodynamics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the example simulation:

```bash
python run_simulation.py
```

This will:
1. Initialize the simulation environment
2. Run a demonstration episode with 5 cables
3. Execute 100 training episodes
4. Display performance statistics

### Using the Simulation in Your Code

```python
from src.simulation.environment import SimulationEnvironment
from src.simulation.cable import Cable

# Create simulation environment
env = SimulationEnvironment()

# Add cables to conveyor belt
for i in range(5):
    cable = Cable()
    env.add_cable_to_belt(cable, x_position=i * 0.4)

# Run simulation episode
results = env.run_episode(num_cables=5, max_steps=10000)

print(f"Success rate: {results['success_rate']:.1%}")
```

## Architecture

### Project Structure

```
Luganodynamics/
├── src/
│   ├── simulation/        # Core simulation components
│   │   ├── cable.py       # Cable object with flexible properties
│   │   ├── conveyor.py    # Conveyor belt simulation
│   │   ├── camera.py      # RGB-D camera simulation
│   │   └── environment.py # Main simulation environment
│   ├── vision/            # Computer vision components
│   │   └── cable_detector.py  # Cable detection and pose estimation
│   ├── robot/             # Robot control components
│   │   └── gripper.py     # Robotic hand/gripper controller
│   └── utils/             # Utility functions
│       └── config.py      # Configuration parameters
├── tests/                 # Unit tests
├── run_simulation.py      # Example simulation script
└── requirements.txt       # Python dependencies
```

### Components

#### 1. Cable Object (`src/simulation/cable.py`)
- Represents cables with connectors on the conveyor
- Models flexible, elongated objects that can curve or twist
- Provides grasp pose calculation for robot pickup

#### 2. Conveyor Belt (`src/simulation/conveyor.py`)
- Simulates moving conveyor belt
- Manages cable placement and movement
- Tracks cables at pickup position

#### 3. RGB-D Camera (`src/simulation/camera.py`)
- Simulates overhead camera with depth sensing
- Generates RGB and depth images
- Provides camera calibration parameters

#### 4. Cable Detector (`src/vision/cable_detector.py`)
- Detects cables in RGB-D images
- Estimates 3D pose of detected cables
- Computes optimal grasp poses

#### 5. Robotic Hand (`src/robot/gripper.py`)
- Controls gripper position and orientation
- Manages grasp sequences
- Provides inverse kinematics (simplified)

#### 6. Simulation Environment (`src/simulation/environment.py`)
- Integrates all components
- Manages simulation loop
- Collects training data and metrics

## Configuration

Simulation parameters can be adjusted in `src/utils/config.py`:

- **Conveyor settings**: Length, width, speed
- **Cable properties**: Dimensions, flexibility
- **Camera settings**: Position, field of view, resolution
- **Robot parameters**: Reach, speed, gripper width
- **Training settings**: Episodes, cables per episode

## Training Approach

The simulation supports a **Simulation-First** approach:

1. **Data Collection**: Run episodes to collect RGB-D images and successful grasp poses
2. **Model Training**: Train neural networks to predict grasp poses from images
3. **Validation**: Test trained models in simulation with various cable configurations
4. **Transfer**: Deploy trained models to real robot hardware

### Key Advantages

- **Safe Training**: No risk to hardware during learning
- **Rapid Iteration**: Fast simulation enables extensive testing
- **Controlled Variation**: Systematically test edge cases
- **Data Abundance**: Generate large training datasets easily

## Problem Statement

The cable with its connector lies on a moving conveyor belt, from which the robot must perform a reliable pickup. From a localization perspective, the connector on the conveyor is an elongated object whose full 3D shape is partly flexible, as the cable may be slightly curved or twisted when it arrives.

**Solution**: The system captures RGB-D images from a calibrated overhead camera and estimates a stable grasp pose that captures the main orientation and position of the connector-cable assembly at the moment of pickup, without requiring complete rigid reconstruction.

## Future Enhancements

- [ ] Integration with PyBullet for physics-based simulation
- [ ] Neural network training pipeline
- [ ] Advanced vision algorithms (deep learning-based pose estimation)
- [ ] Domain randomization for sim-to-real transfer
- [ ] Support for multiple cable types and shapes
- [ ] Real-time visualization
- [ ] ROS integration for real robot deployment

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is open source and available under the MIT License.

## Contact

For questions or feedback, please open an issue on GitHub.