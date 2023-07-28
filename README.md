# Training and Analysis for Multitask RNNs

This project consists of training various RNNs on a collection of tasks and then analyzing the trained networks. The RNN model is defined by the equation `h(t) = (1 - alpha) * h(t-1) + alpha * r(W_in * u(t-1) + W_rec * h(t-1) + b)`, where `alpha = deltaT/tau` is a hyperparameter, `tau` is a time constant, and `r` is the chosen activation function.

The tasks to train the networks on are named "delay-go", "delay-anti", "delay-match-to-sample", and "delay-anti-match-to-sample". Each RNN is trained on a subset of these tasks.

## Tasks

Each task in this project has a defined structure which includes inputs, outputs, periods, and a grace period. The inputs and outputs are binary vectors, and the periods are stages of the task which last for a fixed number of time steps. During the grace period, the network is not penalized for incorrect outputs.

- **Delay-Go**: This task consists of four periods: "fixation", "stimulus on", "delay", and "go". The network should respond to the stimulus that was active during the "stimulus on" period during the "go" period.
- **Delay-Anti**: Similar to the Delay-Go task, but during the "go" period, the network should respond with the opposite stimulus that was active during the "stimulus on" period.
- **DelayGoContinuous**: A variation of the Delay-Go task. Instead of the stimulus being a binary value, it is a float $\theta$ between $0$ and $2\pi$. The input and output tensors encode the sine and cosine of the stimulus rather than a one-hot encoding.
- **DelayAntiContinuous**: Similar to the DelayGoContinuous task, but during the "go" period, the network should respond as though it received the opposite  stimulus ($-\theta$) that was active during the "stimulus on" period.
- **Delay-Match-to-Sample**: Description of Delay-Match-to-Sample task...
- **Delay-Anti-Match-to-Sample**: Description of Delay-Anti-Match-to-Sample task...




## Getting Started

Before running the training and analysis scripts, make sure your Python environment has the required libraries installed. These are:

- PyTorch
- NumPy
- Matplotlib

## Running the Scripts

The project contains one main script:

- `train.py`: Trains the RNNs on the specified tasks and saves the trained networks.
- `find_fixed_points.py`: Finds fixed points of the network for given task details and saves the results.

There are also notebooks:

- `analysis.ipynb`: Plots the behavior of a network and analyzes its fixed point structure.
- `task.ipynb`: Provides visualization of the task structure for each task.

## Project Structure

- `train.py`: The main training script.
- `find_fixed_points.py`: Script to find fixed points of the network for given task details.
- `models/`: Directory to store the trained RNN models.
  - `<model_name>/`: Directory for each model, containing the files specific to that model.
    - `hyperparameters.json`: JSON file containing hyperparameters for the model.
    - `performance.json`: JSON file saving the performance metrics over training.
    - `models_checkpoint/`: Directory to store model checkpoints. Each checkpoint is a file named `model_epoch_<epoch>.pt`, representing the model's state at the specified epoch. There are two additional files: `model.pt`, which represents the model state that has achieved the best validation performance so far, and `model_latest.pt`, which represents the most recent model state.
    - `analysis/`: Directory to store analysis data for the model. 
      - `fixed_points/`: Subdirectory that stores fixed point data for each epoch. Each file is named `fixed_points_epoch_<epoch>.pt`, representing the fixed point data at the specified epoch. Each file contains a dictionary with two keys: `"stable_points"` and `"unstable_points"`. The value for each key is a tensor containing the corresponding set of fixed points.
- `utils/`: Directory that contains various utility files.
  - `model.py`: Defines the `RNN` class, which represents the RNN model.
  - `task.py`: Contains utility functions related to tasks.
  - `analysis.py`: Contains utility functions for analysis.
  - `utils.py`: Contains other utility functions.
- `tasks/`: Directory that contains a separate Python file for each task (for example, `delay_go.py`, `delay_anti.py`, etc.). Each file defines a class for the task that inherits from the base `Task` class. The `Task` class is also defined in this directory.


## Customizing the Training

You can customize the tasks to train the RNNs on by modifying the list of tasks in the `train.py` script. Ensure that all tasks in the list have the same number of inputs and outputs.

## Analysis

The analysis involves examining the behavior of the trained network and finding fixed points. The behavior of the network can be visualized using the `analysis.ipynb` notebook. The `find_fixed_points.py` script can be used to find fixed points of the network for specific task details. Fixed point data is saved in the `models/<model_name>/analysis/` directory specific to each model.
