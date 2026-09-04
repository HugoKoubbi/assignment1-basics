import torch 
from einops import rearrange,einsum
import torch.nn as nn
import numpy as np

def data_loading(x, batch_size, context_length, device):
    """
    inputs: x: numpy array, batch_size: int, context_length: int, device={cpu, cuda}
    output: (x,y), x: tensor:  batch_size context_length, y: tensor batch_size context_length
    """

    n = x.size # obtain the number of tokens

    if n-context_length < batch_size:
        raise ValueError('the size of the input entries is not large enough')

    inputs = np.array([ [x[i] for i in range(j,j+context_length)] for j in range(batch_size)])
    outputs = np.array([ [x[i+1] for i in range(j,j+context_length)] for j in range(batch_size)])

    inputs = torch.tensor(inputs, device=device)
    outputs = torch.tensor(outputs,device=device)

    return (inputs,outputs)


def save_checkpoint(model, optimizer, iteration, out):
    """
    save the state model, optimizer, iteration into the file-like object
    """
    dict_model = model.state_dict() # save the dictionnary of the weights state
    dict_optimizer = optimizer.state_dict() # save the dictionnary of the optimizer state
    dict_out={"model": dict_model,
               "optimizer": dict_optimizer,
                 "iteration" : iteration}
    torch.save(dict_out,out)
    return

def load_checkpoint(src, model, optimizer):
    """
    load the model and the optimizer and returns iteration
    """
    dict_out=torch.load(src)  # load the saved dictionnary previously defined

    model.load_state_dict(dict_out["model"]) # load the model dictionnary
    optimizer.load_state_dict(dict_out["optimizer"]) # load the optimizer dictionnary

    return dict_out["iteration"]

