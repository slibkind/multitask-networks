# Training and Analysis for Multitask RNNs

This project consists of training various RNNs on a collection of tasks and then analyzing the trained networks. The RNN model is defined by the equation `h(t) = (1 - alpha) * h(t-1) + alpha * tanh(W_in * u(t-1) + W_rec * h(t-1) + b)`, where `alpha = deltaT/tau` is a hyperparameter, and `tau` is a time constant. 

The tasks to train the networks on are named "delay-go", "delay-anti", "delay-match-to-sample", and "delay-anti-match-to-sample". Each RNN is trained on a subset of these tasks. 

## Tasks

Each task in this project has a defined structure which includes inputs, outputs, periods, and a grace period. The inputs and outputs are binary vectors and the periods are stages of the task which last for a fixed number of time steps. During the grace period, the network is not penalized for incorrect outputs.

- **Delay-Go**: This task consists of four periods: "fixation", "stimulus on", "delay", and "go". The network should respond to the stimulus that was active during the "stimulus on" period during the "go" period.
- **Delay-Anti**: Description of Delay-Anti task...
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

which can be run from the command line as follows:

```bash
python train.py
```

Two notebooks `tasks.ipynb` and `analysis.ipynb` are used for analyzing the tasks and network behavior.

## Project Structure

- `train.py`: The main training script.
- `models/`: Directory to store the trained RNN models.
  - `<model_name>/`: Directory for each model, containing the files specific to that model.
    - `hparams.json`: JSON file containing hyperparameters for the model.
    - `model.pt`: Saved model file for the model.
    - `analysis/`: Directory to store analysis data for the model.
    ...

- `rnn.py`: Defines the `RNN` class, which represents the RNN model.
- `tasks/`: Directory that contains a separate Python file for each task (for example, `delay_go.py`, `delay_anti.py`, etc.). Each file defines a class for the task that inherits from the base `Task` class. The `Task` class is also defined in this directory.
- `utils.py`: Contains utility functions used in training and analysis.

## Customizing the Training

You can customize the tasks to train the RNNs on by modifying the list of tasks in the `train.py` script. Ensure that all tasks in the list have the same number of inputs and outputs.

## About the Analysis

The analysis involves running gradient descent on the speed of the dynamics for fixed inputs, for each trained RNN. The results of the analysis are saved in the `analysis/` directory within the respective model directory.
