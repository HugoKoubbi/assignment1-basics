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

if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    #Add all the hyperparameters in Parser mode
    #Training parameters
    parser.add_argument("--lr", default=1e-3,type=float)
    parser.add_argument("--wd", default=1e-2,type=float)
    parser.add_argument("--betas", default=(0.99, 0.9),type=tuple)
    parser.add_argument("--alpha_max", default=1e-3,type=float)
    parser.add_argument("--alpha_min", default=1e-4,type=float)
    parser.add_argument("--t_w",default=100,type=int)
    parser.add_argument("--t_c",default=1000,type=int)
    parser.add_argument("--max_norm",default=10.0,type=float)
    parser.add_argument("--context_length", default=128,type=int)
    parser.add_argument("--num_layers", default=12,type=int)
    parser.add_argument("--num_heads", default=8,type=int)
    parser.add_argument("--d_model", default=512,type=int)
    parser.add_argument("--d_ff", default=1344,type=int)
    parser.add_argument("--theta", default=10000,type=int)
    parser.add_argument("--vocab_size", default=10000,type=int)
    parser.add_argument('--iterations', default=5000,type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument("--window_size",default=100,type=int)





    parser.add_argument('--Device', default='mps')
    parser.add_argument("--Checkpoint_paths",default='checkpoints',type=str)
    parser.add_argument("--number_tokens",default=12000000,type=int)
    parser.add_argument("--number_tokens_test",default=50000,type=int)
    parser.add_argument("--wandb_project",default='cs336_basics',type=str)
    parser.add_argument("--wandb_entity",default='hugokoubbi',type=str)
    parser.add_argument("--wandb_name",default='baseline',type=str)
    args = parser.parse_args()

    #Store the hyperparameters in variables

    lr=args.lr
    wd=args.wd
    betas=args.betas
    alpha_max=args.alpha_max
    alpha_min=args.alpha_min
    t_w=args.t_w
    t_c=args.t_c
    max_norm=args.max_norm

    context_length=args.context_length
    num_layers=args.num_layers
    num_heads=args.num_heads
    d_model=args.d_model
    d_ff=args.d_ff
    theta=args.theta
    number_tokens=args.number_tokens
    number_tokens_test=args.number_tokens_test
    vocab_size=args.vocab_size

    iterations=args.iterations
    batch_size=args.batch_size
    Device=args.Device
    window_size=args.window_size

    checkpoint_paths=args.Checkpoint_paths
    nb_non_embedding_parameters= compute_non_embedding_parameters(num_layers,d_model,d_ff)

    nb_parameters= compute_parameters(num_layers,d_model,d_ff,vocab_size)
    print(f'Number of parameters in the model: {nb_parameters}')
    print(f'Number of parameters in the model (in Gb): {4*nb_parameters*10**(-9)}')
    print(f'Number of tokens suggested for training: {20*nb_parameters}')
    print(f'Number of tokens for training: {number_tokens}')
    # Obtaining the device to use for training
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
            test_data = test_data.read(number_tokens_test)
            np.save('data/training_tokenized' ,tokenizer.encode(training_data))
            np.save('data/test_tokenized',tokenizer.encode(test_data))

    # Tokenize the data
    np.save('data/training_tokenized' ,tokenizer.encode(training_data))
    np.save('data/test_tokenized',tokenizer.encode(test_data))

    training_tokenized_mm=np.load('data/training_tokenized.npy',mmap_mode='r')
    test_tokenized_mm=np.load('data/test_tokenized.npy',mmap_mode='r')
    iterations=100

    for b in [1,2,4,8,16,32]:
        start = torch.mps.Event(enable_timing=True)
        end = torch.mps.Event(enable_timing=True)

        start.record()
        inputs_train , labels_train = data_loading(training_tokenized_mm,b,context_length,Device)
        # Initialize the transformers
        model = transformers_lm_sdw_sink(vocab_size,context_length,num_layers,d_model,num_heads,d_ff,window_size,rope_theta=theta)
        model.to(Device)
        model_dict=model.parameters()

        opt = adamw(model_dict, lr=lr,betas=betas,eps=1e-5,weight_decay=wd)

        for steps in range(int(iterations)):
            lr=learning_rate_schedule(steps, alpha_max, alpha_min, t_w, t_c)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss=cross_entropy(model(inputs_train),labels_train) # Compute the cross entropy loss
            print(f'Steps: {steps}, Loss: {loss.item()}')
            loss.backward() # compute the gradient
            print(f"Gradient norm before clipping: {torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)}")
            opt.step() # Run optimizer step.

        end.record()
        torch.mps.synchronize()
        print(f'Time taken for batch_size{b}: {start.elapsed_time(end)} ms')
