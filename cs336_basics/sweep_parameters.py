import torch 
from einops import rearrange,einsum
import torch.nn as nn
import numpy as np
from collections.abc import Callable, Iterable
from torch.utils.data import Dataset
from typing import Optional
import math
import argparse
import wandb
import tqdm
import itertools as iterpols

from cs336_basics.tokenizer import *
from cs336_basics.model import *
from cs336_basics.train import *
from cs336_basics.data import *
from cs336_basics.computations_flops import *


def sweep_parameters():
    x_beta_1=np.linspace(0.95,0.999,5)
    x_beta_2=np.linspace(0.9,0.95,5)
    x_lr=np.linspace(1e-5,1e-3,5)
    x_wd=np.linspace(1e-3,1e-2,5)
    x_alpha_max=np.linspace(1e-3,1e-3,1)
    x_alpha_min=np.linspace(1e-5,1e-3,1)
    x_max_norm=np.linspace(0.1,1.0,1)
    x_tw=np.linspace(10,100,1)
    x_tc=np.linspace(100,1000,1)
    iterations=10
    
    dict_sweep={"beta_1":x_beta_1,"beta_2":x_beta_2,"lr":x_lr,"wd":x_wd,"alpha_max":x_alpha_max,"alpha_min":x_alpha_min,"max_norm":x_max_norm,"tw":x_tw,"tc":x_tc}

    loss_training=np.zeros([len(x_beta_1),len(x_beta_2),len(x_lr),len(x_wd),len(x_alpha_max),len(x_alpha_min),len(x_max_norm),len(x_tw),len(x_tc)])
    for (i_beta1,i_beta2,i_lr,i_wd,i_alpha_max,i_alpha_min,i_max_norm,i_tw,i_tc) in iterpols.product(range(len(x_beta_1)),range(len(x_beta_2)),range(len(x_lr)),range(len(x_wd)),range(len(x_alpha_max)),range(len(x_alpha_min)),range(len(x_max_norm)),range(len(x_tw)),range(len(x_tc))):
        beta1=x_beta_1[i_beta1]
        beta2=x_beta_2[i_beta2]
        lr=x_lr[i_lr]
        wd=x_wd[i_wd]
        alpha_max=x_alpha_max[i_alpha_max]
        alpha_min=x_alpha_min[i_alpha_min]
        max_norm=x_max_norm[i_max_norm]
        tw=x_tw[i_tw]
        tc=x_tc[i_tc]
        loss_training[i_beta1,i_beta2,i_lr,i_wd,i_alpha_max,i_alpha_min,i_max_norm,i_tw,i_tc]=train_model(beta1,beta2,lr,wd,alpha_max,alpha_min,max_norm,tw,tc,iterations=iterations)

    return loss_training,dict_sweep

def finding_optimal_parameters(loss_training,dict_sweep):
    min_index=np.unravel_index(np.argmin(loss_training, axis=None), loss_training.shape)
    optimal_parameters={key: dict_sweep[key][min_index[i]] for i,key in enumerate(dict_sweep.keys())}
    return optimal_parameters

def train_model(beta1,beta2,lr,wd,alpha_max,alpha_min,max_norm,t_w,t_c,iterations=50):
    # Define the hyperparameters for training
    context_length=128
    num_layers=8
    num_heads=4
    d_model=128
    d_ff= 192
    theta=10000
    vocab_size=10000
    batch_size=1
    Device='mps'
    Checkpoint_paths='checkpoints'
    number_tokens=30000
    # Initialize the model and train it
    if torch.backends.mps.is_available():
        Device = torch.device("mps")
    else:
        Device = torch.device("cpu")

    print("Using:", Device)
    
    # Treat the data
    # Without memmap
    with open('data/TinyStoriesV2-GPT4-train.txt','r', encoding="utf-8") as training_data:
        with open('data/TinyStoriesV2-GPT4-valid.txt','r', encoding="utf-8") as test_data:
            # Prepare the tokenizer 

            vocab,merges = train_bpe('data/TinyStories_downscaling.txt',vocab_size,['<|endoftext|>'])
            tokenizer = BPETokenizer(vocab,merges,['<|endoftext|>'])
            print(f'Vocabulary size: {len(vocab)}') 
            print(f'Merges size: {len(merges)}')
            print(f'Tokenizer initialized.')
            # Tokenize the data
            training_data = training_data.read(number_tokens)
            np.save('data/training_tokenized' ,tokenizer.encode(training_data))

    # Tokenize the data
    np.save('data/training_tokenized' ,tokenizer.encode(training_data))

    training_tokenized_mm=np.load('data/training_tokenized.npy',mmap_mode='r')

    # Train the model
    model = transformers_lm(vocab_size,context_length,num_layers,d_model,num_heads,d_ff,rope_theta=theta)
    model.to(Device)
    model_dict=model.parameters()

    betas=(beta1,beta2)
    opt = adamw(model_dict, lr=lr,betas=betas,eps=1e-5,weight_decay=wd)

    for steps in range(int(iterations)):
        inputs_train , labels_train = data_loading(training_tokenized_mm,batch_size,context_length,Device)
        lr=learning_rate_schedule(steps, alpha_max, alpha_min, t_w, t_c)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss=cross_entropy(model(inputs_train),labels_train) # Compute the cross entropy loss
        print(f'Steps: {steps}, Loss: {loss.item()}')
        loss.backward() # compute the gradient
        print(f"Gradient norm before clipping: {torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)}")
        opt.step() # Run optimizer step.
    return loss.item()

loss_training,dict_sweep=sweep_parameters()
print(f"Optimal parameters: {finding_optimal_parameters(loss_training,dict_sweep)}")