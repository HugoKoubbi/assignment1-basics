import torch 
from einops import rearrange,einsum
import torch.nn as nn
import numpy as np


class linear(nn.Module):
    """
    Apply a linear transformation
    args: d_in, d_out,
    """
    def __init__(self, in_features, out_features,device=None,dtype=None):

        #super sert à construire la structure interne du module.

        super().__init__()

        #W: out_features x in_feature, initialized with a certain cutoff

        self.weight=nn.Parameter(torch.empty((out_features,in_features)),requires_grad=True)


        torch.nn.init.trunc_normal_(self.weight,
                                    mean=0,
                                    std=2/(in_features+out_features), 
                                    a=-6/(in_features+out_features),
                                    b=6/(in_features+out_features))
    def forward(self,x):

    #Apply x :torch.tensor(... d_in) -> Wx :torch.tensor (d_out)

        return(einsum( x, self.weight.T, '... d_in, d_in d_out -> ... d_out'))

class embedding(nn.Module):
    """ 
    Apply an embedding layer
    args, num_embeddings: integers, embedding_dim: integer, device, dtype
    param W
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight=nn.Parameter(torch.empty((num_embeddings,embedding_dim)),requires_grad=True)

        torch.nn.init.trunc_normal_(self.weight,
                                    mean=0,
                                    std=1.,
                                    a=-1.,
                                    b=1.)
    def forward(self,tokens_ids):
    # Apply the embedding layer, cette function permet d'appliquer la fonction à chacun des tokens
        return self.weight[tokens_ids]

class rmsnorm(nn.Module):
    """ 
    Apply RMSnorm
    params: epsilon: float, gain : float, d_model
    Normalize
    """
    def __init__(self, d_model, epsilon=1e-5,device=None,dtype=None):
        super().__init__()
        self.d_model=d_model
        self.gain=nn.Parameter(torch.empty(d_model))
        self.epsilon=torch.tensor(epsilon)

        torch.nn.init.normal_(self.gain)

    def forward(self,x):
        in_dtype=x.dtype

        x=x.to(torch.float32)
        rms=torch.sqrt(1/self.d_model * torch.sum(x*x,dim=-1,keepdim=True)+self.epsilon)
        res=x*self.gain/rms

        return res.to(in_dtype)


class positionwise_feedforward(nn.Module):
    """
    Apply the MLP part
    params: d_ff: int, d_model: int, 
    attribute: W_1,W_3 : tensor (d_ff,d_model)  , W_2 : tensor (d_model,d_ff)
    """
    def __init__(self,d_model,d_ff,device=None,dtype=None):

        super().__init__()

        self.w1=nn.Parameter(torch.empty(d_ff,d_model))
        self.w2=nn.Parameter(torch.empty(d_model,d_ff))
        self.w3=nn.Parameter(torch.empty(d_ff,d_model))

        self.d_model=d_model
        self.d_ff=d_ff

    def forward(self,x):
        if (self.d_ff-8/3*self.d_model)**2 > 1000*self.d_model:
            raise ValueError('d_ff is not approximataly 8/3 d_model')
        if (self.d_ff % 64)!=0:
            raise ValueError('d_ff is not a multiple of 64')
        
        w_1=einsum(x,self.w1.T,'... d_model, d_model d_ff-> ... d_ff')

        SiLU=torch.sigmoid(w_1)* w_1
        w_3=einsum(x,self.w3.T,'... d_model, d_model d_ff->... d_ff')

        z=w_3*SiLU

        x=einsum(z,self.w2.T,'... d_ff, d_ff d_model->... d_model')
        return x


class RotaryPositionalEmbedding(nn.Module):
    """
    Apply RoPE 
    params: theta: float, d_k: int, max_seq_len: int
    """
    def __init__(self,theta,d_k,max_seq_len,device=None):
        super().__init__()

        self.theta=theta
        self.d_k=d_k

        if d_k%2!=0:
            raise IndexError('The dimension should be even')
        
        self.max_seq_len=max_seq_len
        inv_freq=1/theta**(2/d_k)

        self.register_buffer("M",
            torch.from_numpy(np.array([

            [[[np.cos(i/inv_freq**k), -np.sin(i/inv_freq**k)],
             [np.sin(i/inv_freq**k), np.cos(i/inv_freq**k)]]

                for k in range(d_k/2)
            ]
                for i in range(max_seq_len)
                ])),
            persistent=False)
        
    def forward(self,x,token_positions):
        #Ici l'astuce c'est de se rendre compte qu'on peut faire le produit par bloc si on découpe le vecteur par deux. Donc pas besoin d'utiliser la matrice pleine (avec que des 0)
        x=rearrange(x,'... n (d pair) -> ... n d pair',
                    pair=2
                    )
        M=M[token_positions]
        x=einsum(self.M, x,' n d i j, ... n d j -> ... n d i')
        return rearrange(x,'... n d i -> ... n (d i)')

#### implementation codex 
class RotaryPositionalEmbedding_gpt(nn.Module):
    def __init__(
        self,
        theta,
        d_k,
        max_seq_len,
        device=None,
    ):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k must be even")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )

        pair_indices = torch.arange(
            d_k // 2,
            device=device,
            dtype=torch.float32,
        )

        inverse_frequencies = theta ** (
            -2 * pair_indices / d_k
        )

        angles = (
            positions[:, None]
            * inverse_frequencies[None, :]
        )

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        first_row = torch.stack([cos, -sin], dim=-1)
        second_row = torch.stack([sin, cos], dim=-1)

        M = torch.stack(
            [first_row, second_row],
            dim=-2,
        )

        # M shape: (max_seq_len, d_k // 2, 2, 2)
        self.register_buffer("M", M, persistent=False)

    def forward(self, x, token_positions):
        input_dtype = x.dtype

        # (..., sequence_length, d_k)
        # -> (..., sequence_length, d_k // 2, 2)
        x_pairs = rearrange(
            x,
            "... n (d pair) -> ... n d pair",
            pair=2,
        )

        # Select a rotation matrix for each token position.
        rotations = self.M[token_positions]
        rotations = rotations.to(dtype=input_dtype)

        # rotations: (..., n, d, 2, 2)
        # x_pairs:  (..., n, d, 2)
        rotated = einsum(
            rotations,
            x_pairs,
            "... n d i j, ... n d j -> ... n d i",
        )

        return rearrange(
            rotated,
            "... n d pair -> ... n (d pair)",
        )
    
def softmax(x,dim):

    shifted= x -torch.amax(x,
                           dim=dim,
                           keepdim=True
                           )

    exponentials=torch.exp(shifted)

    softmax= exponentials/torch.sum(exponentials,
                                    dim=dim,
                                    keepdim=True)


    return  softmax 

def scaled_dot_product_attention(key,query,values,mask):
    #query,key: tensor : b _ n d_k
    #value: tensor : b _ n d_v
    # return attention 

    d = query.shape[-1]
    scores = einsum(query,key,' b ... i q , b ... j q -> b ... i j')

    if mask is not None:
        scores = scores.masked_fill(
                        ~mask,
                        float("-inf"),
                        )    
        
    scores = softmax(scores/np.sqrt(d),dim=-1)
    output = einsum(scores,values,'b ... i j, b ... j d -> b ... i d')

    # à la fin, ce n'est que la matrice s_{ij}=q_{i}^{T}k_{j} 

    return output


class multihead_self_attention_dumb(nn.Module):

    """ 
    Parameters: num_heads: int, d_model: int, 
    Attributes: W_o: tensor , W_v, W_q, W_k
    """

    def __init__(self, d_model, num_heads):

        super().__init__()

        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=int(d_model/num_heads)
        self.d_v=int(d_model/num_heads)

        self.W_o=nn.Parameter(torch.empty(d_model, num_heads* int(d_model/num_heads) ))
        torch.nn.init.normal_(self.W_o)

        self.W_q=nn.Parameter(torch.empty(num_heads* int(d_model/num_heads,d_model) ) )
        torch.nn.init.normal_(self.W_q)

        self.W_k=nn.Parameter(torch.empty(num_heads* int(d_model/num_heads) , d_model))
        torch.nn.init.normal_(self.W_k)

        self.W_v=nn.Parameter(torch.empty(num_heads * int(d_model/num_heads), d_model))
        torch.nn.init.normal_(self.W_v)

        #### Here the smart idea is -> consider a  biggest matrix and then slice it using einsum operations

    
    def forward(self, x):

        ## x: tensor ... , seq_len, d_model  

        W_q=rearrange(self.W_q,'(h d_1) d_2 -> h d_1 d_2' , h=self.num_heads)
        W_k=rearrange(self.W_k,'(h d_1) d_2 -> h d_1 d_2' , h=self.num_heads)
        W_v=rearrange(self.W_v,'(h d_1) d_2 -> h d_1 d_2' , h=self.num_heads)

class multihead_self_attention_wo_rope(nn.Module):


    """ 
    Parameters: num_heads: int, d_model: int, 
    Attributes: W_o: tensor , W_v, W_q, W_k
    """

    def __init__(self, d_model, num_heads, max_seq_len=1024, device=None, dtype=None):

        super().__init__()

        self.d_model=d_model
        self.num_heads=num_heads

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )
        
        self.d_k=int(d_model/num_heads)
        self.d_v=int(d_model/num_heads)
    
        self.o_proj_weight = linear(d_model, num_heads* int(d_model/num_heads))

        self.q_proj_weight = linear(num_heads * int(d_model/num_heads) ,d_model )

        self.k_proj_weight = linear(num_heads* int(d_model/num_heads) , d_model)

        self.v_proj_weight = linear(num_heads * int(d_model/num_heads), d_model)
        self.max_seq_len=max_seq_len
        #### Here the smart idea is to biggest matrix and then slice it using einsum operations
    
    def forward(self, x):

        seq_len = self.max_seq_len
        query = self.q_proj_weight(x)
        key = self.k_proj_weight(x)
        values = self.v_proj_weight(x)


        #query = rearrange(query, '... seq_len (h d_h)  -> ... h seq_len d_h ' , h=self.num_heads)
        #key = rearrange(key,'... seq_len (h d_h)  -> ...  h seq_len d_h' , h=self.num_heads)
        #values=rearrange(values,'... seq_len (h d_h)  -> ...  h seq_len d_h' , h=self.num_heads)

        mask=torch.tril(
            torch.ones(seq_len, seq_len),
            diagonal=0
        ).bool()

        attn=scaled_dot_product_attention(key,query,values,mask)
        #attn=rearrange(attn,'...  h seq_len d_h ->... seq_len (h d_h)' , h=self.num_heads)
        attn=self.o_proj_weight(attn)

        return attn

class multihead_self_attention(nn.Module):

    """ 
    Parameters: num_heads: int, d_model: int, 
    Attributes: W_o: tensor , W_v, W_q, W_k
    """

    def __init__(self, d_model, num_heads, max_seq_len=1024, rope_theta=10000, device=None, dtype=None):

        super().__init__()

        self.d_model=d_model
        self.num_heads=num_heads

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )
        
        self.d_k=int(d_model/num_heads)
        self.d_v=int(d_model/num_heads)
    
        self.o_proj_weight = linear(d_model, num_heads* int(d_model/num_heads))

        self.q_proj_weight = linear(num_heads * int(d_model/num_heads) , d_model)

        self.k_proj_weight = linear(num_heads* int(d_model/num_heads) , d_model)

        self.v_proj_weight = linear(num_heads * int(d_model/num_heads), d_model)
        self.rope_theta=rope_theta

        self.rope=RotaryPositionalEmbedding_gpt(rope_theta,self.d_k,max_seq_len)

        #### Here the smart idea is to biggest matrix and then slice it using einsum operations
    
    def forward(self, x):

        seq_len = x.shape[-2]

        query = self.q_proj_weight(x)
        key = self.k_proj_weight(x)
        values = self.v_proj_weight(x)

        query = rearrange(query, '... seq_len (h d_h)  -> ... h seq_len d_h ' , h=self.num_heads)
        key = rearrange(key,'... seq_len (h d_h)  -> ...  h seq_len d_h' , h=self.num_heads)
        values=rearrange(values,'... seq_len (h d_h)  -> ...  h seq_len d_h' , h=self.num_heads)

        #Apply RoPE 
        token_positions = torch.arange(
                seq_len,
                device=x.device,
        )
            
        query=self.rope(query,token_positions)
        key=self.rope(key,token_positions)


        mask=torch.tril(
            torch.ones(seq_len, seq_len),
            diagonal=0
        ).bool()

        print(mask)

        ### pour le softmax, il faut des queries et keys de la forme b ... n d_v, on considere les tetes comme dans le batch

        attn=scaled_dot_product_attention(key,query,values,mask) # renvoit un vecteur  ..., num_heads, seq_len, d_head
        attn=rearrange(attn,'...  h seq_len d_h ->... seq_len (h d_h)' , h=self.num_heads)
        attn=self.o_proj_weight(attn)

        return attn

class Transformer_block_standard(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,max_seq_len=1024,rope_theta=100):

        super().__init__()

        self.attn=multihead_self_attention(d_model,num_heads,max_seq_len,rope_theta=rope_theta)
        self.ffn=positionwise_feedforward(d_model,d_ff)
        self.rms=rmsnorm(d_model)

    def forward(self,x):
            h1=self.attn(self.rms(x))
            h2=x+h1
            h3=self.rms(h2)
            h4=h2+self.ffn(h3)
            return h4

class Transformer_block_residual(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,max_seq_len=1024,rope_theta=100,depth=32,alpha=1.):

        super().__init__()

        self.attn=multihead_self_attention(d_model,num_heads,rope_theta=rope_theta)
        self.ffn=positionwise_feedforward(d_model,d_ff)
        self.rms=rmsnorm(d_model)

        #Residual scaling
        self.depth=depth
        self.alpha=alpha


    def forward(self,x):
            h1=self.attn(self.rms(x))
            h2=x+h1 * self.depth **(self.alpha)
            h3=self.rms(h2)
            h4=h2+self.ffn(h3)*self.depth **(self.alpha)
            return h4

class transformers_lm(nn.Module):
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

        self.rmsnorm=rmsnorm(d_model)
        self.ln_final=rmsnorm(d_model)

    # Here we used Module List to have a nn.parameter that can contains a list, and then each of the layer is considered as an element of the list.
        self.layers=nn.ModuleList([ 
        Transformer_block_standard(d_model,num_heads,d_ff,max_seq_len=context_length,rope_theta=rope_theta)
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

class transformers_lm_muP(nn.Module):
    def __init__(self,vocab_size,context_length, num_layers,d_model,num_heads,d_ff,alpha):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers


        self.token_embeddings=embedding(vocab_size,d_model)
        self.lm_head=linear(d_model,vocab_size)

        self.rmsnorm=rmsnorm(d_model)
        self.ln_final=rmsnorm(d_model)

    # Here we used Module List to have a nn.parameter that can contains a list, and then each of the layer is considered as an element of the list.
        self.layers=nn.ModuleList([ 
        Transformer_block_residual(d_model,num_heads,d_ff,max_seq_len=context_length,depth=num_layers,alpha=alpha)
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
            