# Luganodynamics

## Robotic Cable Pickup Simulation

A Python simulation for training a robotic hand to pick up flexible cables from a conveyor belt using a **Simulation-First** approach.

### Overview

This simulation models:
- **Flexible cables** as elongated objects that can be curved or twisted
- **Conveyor belt** moving cables through a pickup zone
- **Robotic hand/gripper** that computes optimal grasp poses
- **Training environment** for developing and evaluating grasping strategies

The simulation focuses on computing stable grasp poses that capture the orientation and position of cables at the moment of pickup, accounting for their flexible and variable shapes.

### Features

- **Cable Modeling**: Flexible cable representation with variable curvature and twist
- **Grasp Pose Computation**: Automatic calculation of optimal grasp position, orientation, and approach vector
- **Simulation Environment**: Complete environment integrating cables, conveyor, and robotic hand
- **Visualization**: 2D and 3D visualization of simulation states
- **Training Framework**: Built-in training loop with metrics tracking
- **Extensible Design**: Modular architecture for easy customization

### Installation

1. Clone the repository:
```bash
git clone https://github.com/danutaparaficz2/Luganodynamics.git
cd Luganodynamics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

Run the quick start example:
```bash
python examples/quickstart.py
```

This will:
1. Create a simulation environment
2. Spawn cables on the conveyor belt
3. Attempt to grasp cables as they pass through the pickup zone
4. Generate a visualization of the results

### Usage Examples

#### Basic Simulation

```python
from src.simulation import SimulationEnvironment
from src.visualization import SimulationVisualizer

# Create simulation
sim = SimulationEnvironment(
    conveyor_length=2.0,
    conveyor_width=0.5,
    conveyor_speed=0.1
)

# Spawn cables
for _ in range(3):
    sim.spawn_random_cable()

# Run simulation
for _ in range(100):
    cables_in_zone = sim.conveyor.get_cables_in_pickup_zone(
        sim.pickup_zone_center,
        sim.pickup_zone_radius
    )
    action = 'grasp' if cables_in_zone else None
    sim.step(action)

# Visualize
visualizer = SimulationVisualizer(sim)
visualizer.plot_2d_snapshot(save_path='result.png')
```

#### Training

```python
from src.simulation import SimulationEnvironment
from src.training import SimpleTrainer

# Create simulation and trainer
sim = SimulationEnvironment()
trainer = SimpleTrainer(sim)

# Train over multiple episodes
episode_stats = trainer.train(num_episodes=20)

# Evaluate
eval_stats = trainer.evaluate(num_episodes=5)
```

### Running Demos

The `examples/` directory contains demonstration scripts:

**Full Demo** (includes visualization and training):
```bash
python examples/demo.py
```

This runs:
- Basic simulation with step-by-step visualization
- Single episode demonstration
- Training over multiple episodes with metrics

### Project Structure

```
Luganodynamics/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── cable.py             # Cable model (flexible elongated object)
│   ├── conveyor_belt.py     # Conveyor belt model
│   ├── robotic_hand.py      # Robotic hand with grasp pose computation
│   ├── simulation.py        # Main simulation environment
│   ├── visualization.py     # Visualization tools
│   └── training.py          # Training framework
├── examples/
│   ├── demo.py              # Full demonstration script
│   └── quickstart.py        # Quick start example
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Key Concepts

#### Cable Representation
Cables are modeled as a series of connected points forming a curve. Each cable has:
- Length and number of segments
- Position and orientation in 3D space
- Variable curvature and twist parameters

#### Grasp Pose
A grasp pose consists of:
- **Position**: 3D coordinates where to grasp
- **Orientation**: Direction vector of the cable at grasp point
- **Approach Vector**: Direction from which the gripper approaches
- **Grip Width**: Required gripper opening width
- **Confidence**: Score indicating grasp stability (0-1)

#### Simulation-First Approach
The simulation environment allows:
1. Rapid iteration and testing of grasping strategies
2. Collection of training data without physical hardware
3. Evaluation of different cable configurations
4. Metrics tracking for performance analysis

### Extending the Simulation

The modular design allows easy extension:

**Custom Cable Shapes**:
```python
from src.cable import Cable
cable = Cable(length=1.0)
cable.generate_shape(curvature=3.0, twist=np.pi/2)
```

**Custom Grasp Strategies**:
Inherit from `RoboticHand` and override `compute_grasp_pose()`:
```python
class CustomRoboticHand(RoboticHand):
    def compute_grasp_pose(self, cable):
        # Your custom grasp computation
        pass
```

**Integration with RL Libraries**:
The `SimulationEnvironment` provides:
- `get_state_for_training()`: State vector for neural networks
- `step(action)`: Standard RL step interface
- Episode-based training with metrics

### Requirements

- Python 3.7+
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0

### License

This project is available for use and modification.

### Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

### Contact

For questions or suggestions, please open an issue on GitHub.
