import torch 
from einops import rearrange,einsum
import torch.nn as nn
import numpy as np
from collections.abc import Callable, Iterable
from torch.utils.data import Dataset
from typing import Optional
import math
from model import transformers_lm,transformers_lm_muP
from data import data_loading,save_checkpoint,load_checkpoint
from tokenizer import tokenizer
import argparse

def cross_entropy(logits,target):
    """ 
    inputs: logits : tensor _ vocab_size, target: int \in [0,vocab_size-1]
    """
    logits_tilted=logits-torch.amax(logits,dim=-1,keepdim=True)
    target=rearrange(target, 'b-> b 1')
    logits_aux=torch.gather(logits_tilted,1,target)

    #Gathering tensors shaped (B,V) and (B,1) produces (B,1).
    #With dim 0, output[i,0] comes from Z[index[i,0],0].
    #With dim 1, output[i,0] comes from Z[i,index[i,0]].
    
    loss=torch.mean(
        -logits_aux+torch.log(torch.sum(torch.exp(logits_tilted),dim=-1)),
                    dim=0)

    return loss


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):

        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.

            for p in group["params"]:

                # if no gradient then skip            
                if p.grad is None:
                    continue

                # if gradient then perform SGD

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.

                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.

                state["t"] = t + 1 # Increment iteration number.

        return loss

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)

for t in range(100):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.

for lr in [1e1,1e2,1e3,1e4]:
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr)
    for t in range(100):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(f'Loss for lr={lr}: {loss.cpu().item()}')
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.


##### Dichotomy search for the blowing-up learning rate
a=1000
b=100
for i in range(10):
    lr=(a+b)/2
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr)
    for t in range(20):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(f'Loss for lr={lr}: {loss.cpu().item()}')
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.
    if loss.cpu().item()>100:
        a=lr
    else:
        b=lr


class adamw(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.99,0.9), eps=1e-5, wd=1e-2):

        if lr<0:
            raise ValueError(f'Invalid learning rate:{lr}')

        beta1,beta2=betas

        if beta1>=1 or beta1<0:
            raise ValueError(f'Invalid beta1 :{beta1}')

        if beta2>=1 or beta1<0:
            raise ValueError(f'Invalid beta2 :{beta2}')

        if eps<=0:
            raise ValueError(f'Invalid epsilon:{eps}')

        if wd<=0:
            raise ValueError(f'Invalid epsilon:{wd}')


        defaults = {"lr": lr, "beta1": beta1, "beta2": beta2, "eps": eps, "wd": wd}

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):

        loss = None if closure is None else closure()


        # each parameters get its own lr, beta1, beta 2, epsilon, wd

        for group in self.param_groups:
            lr = group["lr"]     # Get the learning rate.
            beta1 = group["beta1"] # get the beta 1 variable
            beta2 = group["beta2"] # get the beta 2 variable
            eps = group["eps"]     # Get the epsilon variable 
            wd = group["wd"]    # get the weight decay variable


            for p in group["params"]:

                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p. 
                t = state.get("t", 1) # Get iteration number from the state, or 0.
                m = state.get("m", torch.zeros_like(p)) # Get the tensor m from the state, or
                v = state.get("v", torch.zeros_like(p)) # Get the tensor v from the state, or



                grad = p.grad.data # Get the gradient of loss with respect to p.
                alpha = lr * ( 1-beta2**(t) )**(1/2) / ( 1-beta1**(t) ) # Compute the adjusted lr at iteration t
                p.data = p.data -lr * wd * p.data # Apply weight decay
                m = beta1 * m +(1-beta1) * grad # Update the first moment estimate
                v = beta2 * v + (1-beta2) *grad**2 # Update the second moment estimate
                p.data -= alpha *m /( torch.sqrt(v)+ eps) # Update weight tensor in-place.

                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m # Increment the tensor m
                state["v"] = v # increment the tensor v

        return loss
        

def learning_rate_schedule(t,alpha_max,alpha_min,t_w,t_c):
    if t_c <= t_w:
        raise ValueError('Invalid values for t_c and t_w')
    
    if t < t_w:
        return t/t_w *alpha_max
    
    elif t_w <= t <= t_c:
        return alpha_min +0.5 *(1 + math.cos(math.pi * (t-t_w)/(t_c-t_w) ))*(alpha_max-alpha_min)
    
    else:
        return alpha_min

def gradient_clipping(g, max_norm, eps=1e-6):

    i=0
    for p in g:
        if p.grad is None:
            continue
        else:
            device=p.grad.device
            dtype=p.grad.dtype
            norm=p.grad
            i+=1
            break

    if i==0:
        raise ValueError('No gradient found in the parameters')
    norm=norm.new_zeros((), device=device, dtype=dtype)

    for p in g:
        if p.grad is None:
            continue
        else:
            norm+=torch.norm(p.grad)**2

    norm = torch.sqrt(norm)
    if norm < max_norm:
        return 
    else:
        for p in g:
            if p.grad is None:
                continue
            else:
                p.grad.mul_(max_norm /(eps+norm))
        return 



if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    #Add all the hyperparameters in Parser mode
    #Training parameters
    parser.add_argument("lr", default=1e-3,type=float)
    parser.add_argument("wd", default=1e-2,type=float)
    parser.add_argument("betas", default=(0.9, 0.999),type=tuple)
    parser.add_argument("alpha_max", default=1e-3,type=float)
    parser.add_argument("alpha_min", default=1e-5,type=float)
    parser.add_argument("t_w",default=1000,type=int)
    parser.add_argument("t_c",default=10000,type=int)
    parser.add_argument("max_norm",default=10.0,type=float)

    parser.add_argument("context_length", default=1024,type=int)
    parser.add_argument("num_layers", default=12,type=int)
    parser.add_argument("num_heads", default=12,type=int)
    parser.add_argument("d_model", default=768,type=int)
    parser.add_argument("d_ff", default=3072,type=int)
    parser.add_argument("theta", default=10000,type=int)
    parser.add_argument("vocab_size", default=50257,type=int)

    parser.add_argument('iterations', default=100,type=int)
    parser.add_argument('batch_size', default=2, type=int)
    parser.add_argument('Device', default='cpu')
    parser.add_argument("Checkpoint_paths",default='checkpoints/',type=str)
    
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
    training_data = open('data/TinyStoriesV2-GPT4-train.txt','r', encoding="utf-8")
    test_data = open('data/TinyStoriesV2-GPT4-test.txt','r', encoding="utf-8")

    # Prepare the tokenizer (TBD soon)
    vocab={ "{i}": ord(i) for i in range(256)}
    tokenizer=tokenizer()

    # Tokenize the data
    training_tokenized = np.save('data/' ,tokenizer.encode(training_data))
    test_tokenized = np.save('data/',tokenizer.encode(test_data))

    training_tokenized_mm=np.load('data/training_tokenized',mmap_mode='r')
    test_tokenized_mm=np.load('data/test_tokenized',mmap_mode='r')

    #creating batch_size, inputs,outputs
    inputs_train , labels_train = data_loading(training_tokenized_mm,batch_size,context_length)
    inputs_test , labels_test = data_loading(test_tokenized_mm,batch_size,context_length)

    # Initialize the transformers
    model = transformers_lm(vocab_size,context_length,num_layers,d_model,num_heads,d_ff)

    model_dict=transformers_lm.state_dict

    opt = adamw(model_dict, lr=lr,betas=betas,eps=1e-5,wd=wd)

    for steps in range(iterations):
        #Checkpoints for every 1000 steps
        if steps % 1000 == 0:
            save_checkpoint(model, opt, steps, checkpoint_paths+f'checkpoint_{step}.pt')
        
        #Get the actual learning rate
        lr=learning_rate_schedule(steps, alpha_max, alpha_min, t_w, t_c)
        for p in model.parameters:
            if p.grad is None:
                continue
            p["lr"] = lr

        opt.zero_grad() # Reset the gradients for all learnable parameters.

        output_train=model(inputs_train) # Compute the model outputs
        loss=cross_entropy(output_train,labels_train) # Compute the cross entropy loss
        loss.backward() # compute the gradient

        gradient_clipping(model.parameters(), max_norm=max_norm) # gradient clipping
        
        opt.step() # Run optimizer step.
        

        


        






    

    