import os 
import json
import torch 

from utils.model import MultitaskRNN


def get_model_dir(model_name):
    models_dir = 'models/'
    return os.path.join(models_dir, model_name)


def get_hparams(model_name):
    model_dir = get_model_dir(model_name)
    hparam_path = os.path.join(model_dir, "hparams.json")
    with open(hparam_path, 'r') as f:
        hparams = json.load(f)

    hparams["learning_rate"] = 10**(hparams["lr"]/2)
    hparams["alpha"] = hparams["dt"]/hparams["tau"]
    return hparams

def get_metrics_path(model_name):
    model_dir = get_model_dir(model_name)
    return os.path.join(model_dir, "performance.json")


def get_model_path(model_name, epoch=None, latest=False):
    model_dir = get_model_dir(model_name)
    model_checkpoint_dir = os.path.join(model_dir, "model_checkpoint")
    
    # Check if the directory exists and create it if it doesn't
    if not os.path.isdir(model_checkpoint_dir):
        os.makedirs(model_checkpoint_dir)
    
    if latest:  # Get the latest model
        return os.path.join(model_checkpoint_dir, "model_latest.pt")

    if epoch:  # Get the model of a specific epoch
        return os.path.join(model_checkpoint_dir, f"model_epoch_{epoch}.pt")
    
    return os.path.join(model_checkpoint_dir, "model.pt")


def get_model_checkpoint(model_name, epoch=None, latest=False):
    model_path = get_model_path(model_name, epoch=epoch, latest=latest)
    return torch.load(model_path)

def load_checkpoint(model_name, model, optimizer, epoch=None, latest=False):
    """Load a model and optimizer state from a checkpoint file."""
    checkpoint = get_model_checkpoint(model_name, epoch=epoch, latest=latest)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    
    return model, optimizer, start_epoch

def get_tasks(model_name, epoch=None, latest=False):
    model_checkpoint = get_model_checkpoint(model_name, epoch=epoch, latest=latest)
    return model_checkpoint['tasks']

def get_model(model_name, epoch=None, latest=False):
    model_checkpoint = get_model_checkpoint(model_name, epoch=epoch, latest=latest)
    model_state_dict = model_checkpoint['model_state_dict']
    tasks = model_checkpoint['tasks']

    hparams = get_hparams(model_name)
    
    # Create the MultitaskRNN instance
    rnn = MultitaskRNN(input_size=tasks[0].num_inputs + len(tasks),
                       hidden_size=hparams["num_hidden"],
                       output_size=tasks[0].num_outputs,
                       hparams=hparams)
    rnn.load_state_dict(model_state_dict)
    return rnn, tasks

def get_model_epoch(model_name, epoch=None, latest=False): 
    checkpoint = get_model_checkpoint(model_name, epoch=epoch, latest=latest)
    return checkpoint['epoch']

def get_analysis_path(model_name):
    model_dir = get_model_dir(model_name)
    analysis_path = os.path.join(model_dir, "analysis")

    # Check if the directory exists and create it if it doesn't
    if not os.path.isdir(analysis_path):
        os.makedirs(analysis_path)

    return analysis_path

def get_fixed_point_path(model_name, epoch=None):
    analysis_path = get_analysis_path(model_name)
    fixed_point_path = os.path.join(analysis_path, "fixed_points")

    # Check if the directory exists and create it if it doesn't
    if not os.path.isdir(fixed_point_path):
        os.makedirs(fixed_point_path) 
    
    if epoch == None:
        epoch = get_model_epoch(model_name)

    return os.path.join(fixed_point_path, f"fixed_points_epoch_{epoch}.pt")

def input_to_str(input):
    return str(input.tolist())