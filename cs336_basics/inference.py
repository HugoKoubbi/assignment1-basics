from cs336_basics.tokenizer import *
from cs336_basics.model import *
from cs336_basics.train import *
from cs336_basics.data import *
import wandb 
import argparse
import torch 
from einops import rearrange,einsum
import torch.nn as nn
import numpy as np

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
    parser.add_argument("--context_length", default=256,type=int)
    parser.add_argument("--num_layers", default=6,type=int)
    parser.add_argument("--num_heads", default=4,type=int)
    parser.add_argument("--d_model", default=512,type=int)
    parser.add_argument("--d_ff", default=1344,type=int)
    parser.add_argument("--theta", default=10000,type=int)
    parser.add_argument("--vocab_size", default=10000,type=int)

    parser.add_argument('--iterations', default=5000,type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--Device', default='cpu')
    parser.add_argument("--Checkpoint_paths",default='checkpoints',type=str)
    
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
    vocab_size=args.vocab_size

    iterations=args.iterations
    batch_size=args.batch_size
    Device=args.Device

    checkpoint_paths=args.Checkpoint_paths
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
            training_data = training_data.read(100000)
            test_data = test_data.read(5000)
            np.save('data/training_tokenized' ,tokenizer.encode(training_data))
            np.save('data/test_tokenized',tokenizer.encode(test_data))

    # Tokenize the data
    np.save('data/training_tokenized' ,tokenizer.encode(training_data))
    np.save('data/test_tokenized',tokenizer.encode(test_data))

    model = transformers_lm(vocab_size,context_length,num_layers,d_model,num_heads,d_ff,rope_theta=theta)
    model.to(Device)
    model_dict=model.parameters()
    opt = adamw(model_dict, lr=lr,betas=betas,eps=1e-5,weight_decay=wd)

    load_checkpoint('/Users/hugokoubbi/Developer/assignment1-basics/ checkpoints/run_15_09/checkpointscheckpoint_2000.pt', model, opt)

    print(generate_text(torch.from_numpy(np.array(tokenizer.encode('I will tell a story.'))).unsqueeze(0),model,10.,1.,0.1, 'basic' ,tokenizer))
    if torch.backends.mps.is_available():
        Device = torch.device("mps")
    else:
        Device = torch.device("cpu")

    print("Using:", Device)
    recommended_gib = torch.mps.recommended_max_memory() / (1024**3)
    print(f"Recommended MPS working set: {recommended_gib:.2f} GiB")
