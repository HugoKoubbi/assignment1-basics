import torch 
from einops import rearrange,einsum
import torch.nn as nn
import numpy as np
import numpy as np
import torch


def data_loading(x, batch_size, context_length, device):
    """x : tableau NumPy 1D contenant des identifiants de tokens."""
    if x.ndim != 1:
        raise ValueError("x doit être un tableau 1D")

    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size et context_length doivent être positifs")

    n = x.size
    if n <= context_length:
        raise ValueError("Il faut au moins context_length + 1 tokens")

    # Une position de départ par séquence, tirée avec remise.
    starts = np.random.randint(
        low=0,
        high=n - context_length,
        size=batch_size,
    )

    # [B, 1] + [1, S] donne un tableau d'indices [B, S].
    indices = starts[:, None] + np.arange(context_length)[None, :]

    # On convertit uniquement les batches, pas le corpus entier.
    inputs = np.ascontiguousarray(x[indices], dtype=np.int64)
    outputs = np.ascontiguousarray(x[indices + 1], dtype=np.int64)

    return (
        torch.from_numpy(inputs).to(device),
        torch.from_numpy(outputs).to(device),
    )

def data_loadin_old2(x, batch_size, context_length, device):
    """
    inputs: x: numpy array, batch_size: int, context_length: int, device={cpu, cuda}
    output: (x,y), x: tensor:  batch_size context_length, y: tensor batch_size context_length
    """

    n = x.size # obtain the number of tokens

    if n-context_length < batch_size:
        raise ValueError('the size of the input entries is not large enough')

    bs=[np.random.randint(low=0,high=n-context_length) for i in range(batch_size)]
    inputs = []
    outputs = []
    for j in bs:
        u=x[j:j+context_length+1]
        inputs.append(u[:-1])
        outputs.append(u[1:])
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    inputs = torch.tensor(inputs, device=device)
    outputs = torch.tensor(outputs,device=device)

    return (inputs,outputs)

def data_loading_old(x, batch_size, context_length, device):
    """
    inputs: x: numpy array, batch_size: int, context_length: int, device={cpu, cuda}
    output: (x,y), x: tensor:  batch_size context_length, y: tensor batch_size context_length
    """

    n = x.size # obtain the number of tokens

    if n-context_length < batch_size:
        raise ValueError('the size of the input entries is not large enough')

    bs=[np.random.randint(low=0,high=n-context_length) for i in range(batch_size)]

    inputs = np.array([ [x[i] for i in range(j,j+context_length)] for j in bs])
    outputs = np.array([ [x[i+1] for i in range(j,j+context_length)] for j in bs])

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
    dict_out=torch.load(src, map_location=torch.device('cpu'))  # load the saved dictionnary previously defined

    model.load_state_dict(dict_out["model"]) # load the model dictionnary
    optimizer.load_state_dict(dict_out["optimizer"]) # load the optimizer dictionnary

    return dict_out["iteration"]

