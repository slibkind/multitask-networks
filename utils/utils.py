import os 
import json
import torch 

from utils.model import MultitaskRNN

model_dir = 'models/'

def get_hparams(model_name):
    hparam_path = os.path.join(model_dir, model_name, "hparams.json")
    with open(hparam_path, 'r') as f:
        hparams = json.load(f)

    hparams["learning_rate"] = 10**(hparams["lr"]/2)
    hparams["alpha"] = hparams["dt"]/hparams["tau"]
    return hparams

def get_model_dir(model_name):
    return os.path.join(model_dir, model_name)

def get_model_path(model_name):
    return os.path.join(model_dir, model_name, "model.pt")

def get_model_checkpoint(model_name):
    model_path = get_model_path(model_name)
    return torch.load(model_path)

def get_tasks(model_name):
    model_checkpoint = get_model_checkpoint(model_name)
    return model_checkpoint['tasks']

def get_model(model_name):
    model_checkpoint = get_model_checkpoint(model_name)
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

def get_analysis_path(model_name):
    model_path = get_model_dir(model_name)
    analysis_path = os.path.join(model_path, "analysis")
    return analysis_path

def get_fixed_point_path(model_name, input):
    analysis_path = get_analysis_path(model_name)

    # Convert interpolated_input to string for file naming
    input_str = "_".join([str(int(t.item())) for t in (input * 1000)])
    
    return os.path.join(analysis_path, f'fixed_points_{input_str}.pt')

def get_fixed_points(model_name, input):
    fixed_point_path = get_fixed_point_path(model_name, input)
    return torch.load(fixed_point_path)