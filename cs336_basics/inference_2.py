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
    parser.add_argument("--lr", default=2e-3,type=float)
    parser.add_argument("--wd", default=5e-3,type=float)
    parser.add_argument("--betas", default=(0.9, 0.999),type=tuple)
    parser.add_argument("--alpha_max", default=1e-3,type=float)
    parser.add_argument("--alpha_min", default=1e-4,type=float)
    parser.add_argument("--t_w",default=500,type=int) # choose 1-10% of the run that is in the warmup phase
    parser.add_argument("--t_c",default=5000,type=int) # choose such that the end of training coincides with the end of decay phases
    parser.add_argument("--max_norm",default=10.0,type=float)
    parser.add_argument("--context_length", default=256,type=int)
    parser.add_argument("--num_layers", default=4,type=int)
    parser.add_argument("--num_heads", default=16,type=int)
    parser.add_argument("--d_model", default=512,type=int)
    parser.add_argument("--d_ff", default=1344,type=int)
    parser.add_argument("--theta", default=10000,type=int)
    parser.add_argument("--vocab_size", default=10000,type=int)
    parser.add_argument('--iterations', default=5005,type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--number_tokens_test', default=100000, type=int)
    parser.add_argument('--number_tokens', default=40000000, type=int)
    parser.add_argument('--Device', default='mps')
    parser.add_argument("--Checkpoint_paths",default='checkpoints',type=str)
    parser.add_argument("--Reused_training", default=True, type=bool )
    parser.add_argument("--checkpoint_load", default="checkpoints", type=str)
    parser.add_argument("--training_time_loading", default=0, type=int)
    
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
    number_tokens=args.number_tokens
    number_tokens_test=args.number_tokens_test
    Device=args.Device

    checkpoint_paths=args.Checkpoint_paths
    checkpoint_load="checkpoints/checkpoint_run17M_24000.pt"
    training_time_loading=24000


    nb_non_embedding_parameters= compute_non_embedding_parameters(num_layers,d_model,d_ff)

    nb_parameters= compute_parameters(num_layers,d_model,d_ff,vocab_size)
    print(f'Number of parameters in the model: {nb_parameters}')
    print(f'Number of parameters in the model (in Gb): {4*nb_parameters*10**(-9)}')
    print(f'Number of tokens suggested for training: {20*nb_parameters}')
    print(f'Number of tokens for training: {number_tokens}')
    # Obtaining the device to use for training
    if torch.backends.mps.is_available():
        Device = torch.device("mps")
        print("Using MPS device.")
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
    #np.save('data/training_tokenized' ,tokenizer.encode(training_data))
    #np.save('data/test_tokenized',tokenizer.encode(test_data))

    #training_tokenized_mm=np.load('data/training_tokenized.npy',mmap_mode='r')
    #test_tokenized_mm=np.load('data/test_tokenized.npy',mmap_mode='r')
    #run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    #    entity="koubbihugo-university-paris-dauphine",
    # Set the wandb project where this run will be logged.
    #    project="CS336-assignment",
    # Track hyperparameters and run metadata.
    #    config={
    #    "learning_rate": lr,
    #    "architecture": "Transformer",
    #    "dataset": "TinyStories",
    #    "epochs": iterations,
    #    "batch_size": batch_size,
    #    "context_length": context_length,
    #    "num_layers": num_layers,
    #    "num_heads": num_heads,
    #    "d_model": d_model,
    #    "d_ff": d_ff,
    #    "theta": theta,
    #    "vocab_size": vocab_size,
    #    "max_norm": max_norm,
    #    },
    #)
    with torch.no_grad():
        model = transformers_lm(vocab_size,context_length,num_layers,d_model,num_heads,d_ff,rope_theta=theta)
        model.to(Device)
        model=torch.compile(
            model,
            backend="inductor",
            mode="default",
            dynamic=False,)
        checkpoint = torch.load('checkpoints/checkpoint_run17M_24000.pt', map_location=Device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

    #model_dict=model.parameters()
    #opt = adamw(model_dict, lr=lr,betas=betas,eps=1e-5,weight_decay=wd)
    #opt.load_state_dict(checkpoint['optimizer'])

    #load_checkpoint('checkpoints/checkpoint_16000.pt', model, opt)
 

    print(generate_text(torch.from_numpy(np.array(tokenizer.encode('Once upon a time,'))).unsqueeze(0),model,200,0.01,0.1, 'basic' ,tokenizer))
    print(generate_text(torch.from_numpy(np.array(tokenizer.encode('I will tell a story.  '))).unsqueeze(0),model,200,0.01,0.1, 'basic' ,tokenizer))
    print(generate_text(torch.from_numpy(np.array(tokenizer.encode('Camille is a girl living '))).unsqueeze(0),model,200,0.01,0.1, 'basic' ,tokenizer))
    print(generate_text(torch.from_numpy(np.array(tokenizer.encode('Camille is the most beautiful girl in the world, and she is  '))).unsqueeze(0),model,200,0.01,0.1, 'basic' ,tokenizer))
    print(generate_text(torch.from_numpy(np.array(tokenizer.encode('Camille is the most beautiful girl in the world, and she is  '))).unsqueeze(0),model,500,0.01,0.1, 'basic' ,tokenizer))
    print(generate_text(torch.from_numpy(np.array(tokenizer.encode('Camille is the most beautiful girl in the world, and she is  '))).unsqueeze(0),model,500,0.001,0.1, 'basic' ,tokenizer))
    print(generate_text(torch.from_numpy(np.array(tokenizer.encode('A mushroom called Harry '))).unsqueeze(0),model,500,0.01,0.1, 'basic' ,tokenizer))


    



