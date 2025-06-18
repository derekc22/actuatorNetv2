# Bipedal Robot Torque Prediction and Data Collection Suite

This project implements a full pipeline for simulating, collecting, training, and evaluating a neural network-based torque prediction model for a bipedal robot in the MuJoCo physics engine. The project consists of three main components:

- A physics-based simulated robot model (`biped_simple_final.xml`)
- A neural network architecture and training routine (`actuator_net.py`)
- A comprehensive simulation and data collection system (`data_collecter.py`)

## Directory Structure

```bash
.
├── actuator_net.py              # Neural network model and training script
├── biped_simple_final.xml       # MuJoCo XML model of the biped robot
├── data_collecter.py            # Data collection and evaluation script
├── collected_data/              # Output folder for simulation data
└── trained_models/              # Output folder for trained models
```

## Dependencies

Install the following Python libraries:

```bash
pip install torch mujoco numpy pandas matplotlib
```

Also ensure that you have MuJoCo properly installed and licensed on your system.

## Usage

### 1. Data Collection

Run `data_collecter.py` with appropriate arguments to simulate robot behavior and collect joint torque data.

```bash
python data_collecter.py -c -d -t 10 -n 5
```

Available flags:

- `-c`: Enable data collection mode
- `-s`: Collect shelf data (currently placeholder)
- `-d`: Collect drop data
- `-k`: Collect kickoff data
- `-w`: Collect wall push data
- `-p`: Collect perturbation data
- `-t`: Duration in seconds
- `-n`: Number of trials

The output will be saved to `./collected_data/inverse_kinematics/concatenated_data.pt`

### 2. Model Training

Use the training script to train a neural network that maps 60 input features to 10 joint torques.

```bash
python actuator_net.py
```

This saves the model checkpoint and final weights to `./trained_models/concatenated/`.

### 3. Evaluation

Run the evaluation phase without the `-c` flag:

```bash
python data_collecter.py -t 10
```

The script will:

- Load the trained model
- Simulate robot motion using inverse kinematics
- Compare predicted vs actual torques
- Save plots and pickle files for post-analysis

### 4. Output Files

- `concatenated_data.pt`: Contains training input-output pairs
- `concatenated_model.pth`: Trained model weights
- `concatenated_loss_curve.png`: Training and validation loss over time
- `torque_comparisons/`: Folder with visual comparison of predicted vs actual torques

## Model Architecture

- **Input**: 60 features (10 joints × 3 time steps × 2 values: position error and velocity)
- **Output**: 10 torques (one for each joint)
- **Hidden Layers**: 3 layers, each with 128 neurons and `Softsign` activation

## Robot Model

The `biped_simple_final.xml` describes a humanoid robot with 10 actuated joints. The MuJoCo model includes:

- Sensors for position, velocity, torque
- Actuators with control ranges
- Realistic physical parameters and constraints

## Notes

- The entire system is deterministic for reproducibility
- Sensor and joint mappings are explicitly handled
- Trajectory generation is done with phase-offset sine waves

## License

This project is provided without a license. Please add one before public distribution.

## Contact

This software is maintained internally. For inquiries, please contact the author.
