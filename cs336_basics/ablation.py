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



class PreLayerNorm_Transformers(nn.Module):
    def __init__(self,vocab_size,context_length, num_layers,d_model,num_heads,d_ff,rope_theta):

        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers


        self.token_embeddings=embedding(vocab_size,d_model)
        self.lm_head=linear(d_model,vocab_size)

        #self.rmsnorm=rmsnorm(d_model)
        self.ln_final=rmsnorm(d_model)

    # Here we used Module List to have a nn.parameter that can contains a list, and then each of the layer is considered as an element of the list.
        self.layers=nn.ModuleList([ 
        PostLayer_Transformerblock(d_model,num_heads,d_ff,max_seq_len=context_length,rope_theta=rope_theta)
        for l in range(num_layers)])

    def forward(self,x):

        x=self.token_embeddings(x)

        for layer in self.layers:
            x=layer(x)

        x=self.ln_final(x)        
        x=self.lm_head(x)

        probes=x
            #probes=softmax(x,dim=-1)

        return probes

class PostLayer_Transformerblock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,max_seq_len=1024,rope_theta=100):

        super().__init__()

        self.attn=multihead_self_attention(d_model,num_heads,max_seq_len,rope_theta=rope_theta)
        self.ffn=positionwise_feedforward(d_model,d_ff)
        self.ln1=rmsnorm(d_model)
        self.ln2=rmsnorm(d_model)

    def forward(self,x):
            h1=self.attn(x)
            h2=self.ln1(x+h1)
            h4=self.ln2(h2+self.ffn(h2))
            return h4

class Nope_Transformers(nn.Module):
    def __init__(self,vocab_size,context_length, num_layers,d_model,num_heads,d_ff,rope_theta):

        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers


        self.token_embeddings=embedding(vocab_size,d_model)
        self.lm_head=linear(d_model,vocab_size)

        #self.rmsnorm=rmsnorm(d_model)
        self.ln_final=rmsnorm(d_model)

    # Here we used Module List to have a nn.parameter that can contains a list, and then each of the layer is considered as an element of the list.
        self.layers=nn.ModuleList([ 
        Nope_Transformers_block(d_model,num_heads,d_ff,max_seq_len=context_length,rope_theta=rope_theta)
        for l in range(num_layers)])

    def forward(self,x):

        x=self.token_embeddings(x)

        for layer in self.layers:
            x=layer(x)

        x=self.ln_final(x)        
        x=self.lm_head(x)

        probes=x
            #probes=softmax(x,dim=-1)

        return probes
    
class Nope_Transformers_block(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,max_seq_len=1024,rope_theta=100):
        super().__init__()
        self.attn=multihead_self_attention_wo_rope(d_model,num_heads,max_seq_len,rope_theta=rope_theta)
        self.ffn=positionwise_feedforward(d_model,d_ff)
        self.ln1=rmsnorm(d_model)
        self.ln2=rmsnorm(d_model)

    def forward(self,x):
            h1=self.attn(x)
            h2=self.ln1(x+h1)
            h4=self.ln2(h2+self.ffn(h2))
            return h4    


    