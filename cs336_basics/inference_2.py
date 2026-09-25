import argparse
import torch
import cs336_basics
from cs336_basics import *
from cs336_basics.tokenizer import BPETokenizer
from pathlib import Path
from cs336_basics.model import *
from cs336_basics.train import *
from cs336_basics.data import *

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
    parser.add_argument("--num_layers", default=12,type=int) # Reverted to larger size
    parser.add_argument("--num_heads", default=8,type=int) # Reverted to larger size
    parser.add_argument("--d_model", default=512,type=int) # Reverted to larger size
    parser.add_argument("--d_ff", default=1344,type=int) # Reverted to larger size
    parser.add_argument("--theta", default=10000,type=int)
    parser.add_argument("--vocab_size", default=16384,type=int)
    parser.add_argument('--iterations', default=20000,type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--Device', default='mps')
    parser.add_argument("--Checkpoint_paths",default='checkpoints',type=str)

    args = parser.parse_args([])

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

    # Obtaining the device to use for training
    if torch.backends.mps.is_available():
        Device = torch.device("mps")
    elif torch.cuda.is_available():
      Device = torch.device("cuda")
    else:
        Device = torch.device("cpu")

    print("Using:", Device)
    with open('data/TinyStoriesV2-GPT4-train.txt','r', encoding="utf-8") as training_data:
            with open('data/TinyStoriesV2-GPT4-valid.txt','r', encoding="utf-8") as test_data:
                # Prepare the tokenizer 
    
                vocab,merges = train_bpe('data/TinyStories_downscaling.txt',vocab_size,['<|endoftext|>'])
                tokenizer = BPETokenizer(vocab,merges,['<|endoftext|>'])
                print(f'Vocabulary size: {len(vocab)}') 
                print(f'Merges size: {len(merges)}')
                print(f'Tokenizer initialized.')
                # Tokenize the data
                training_data = training_data.read(500000)
                test_data = test_data.read(5000)
                np.save('data/training_tokenized' ,tokenizer.encode(training_data))
                np.save('data/test_tokenized',tokenizer.encode(test_data))

    # Initialize the transformers
    model = transformers_lm(vocab_size,context_length,num_layers,d_model,num_heads,d_ff,rope_theta=theta)
    model.to(Device)
    model_dict=model.parameters()
    opt = adamw(model_dict, lr=lr,betas=betas,eps=1e-5,weight_decay=wd)

    load_checkpoint('checkpoints/checkpoint_16000.pt', model, opt)


    print(generate_text(torch.from_numpy(np.array(tokenizer.encode('Once upon a time,'))).unsqueeze(0),model,20,1.,0.1, 'basic' ,tokenizer))