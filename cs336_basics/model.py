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
        self.W=nn.Parameter(torch.empty((out_features,in_features)),requires_grad=True)

        torch.nn.init.trunc_normal_(self.W,
                                    mean=0,
                                    std=2/(in_features+out_features), 
                                    a=-6/(in_features+out_features),
                                    b=6/(in_features+out_features))
    def forward(self,x):
    #Apply x :torch.tensor(d_in) -> Wx :torch.tensor (d_out)
        return(einsum( x, self.W.T, '... d_in, d_in d_out -> ... d_out'))

class embedding(nn.Module):
    """ 
    Apply an embedding layer
    args, num_embeddings: integers, embedding_dim: integer, device, dtype
    param W
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.W=nn.Parameter(torch.empty((num_embeddings,embedding_dim)),requires_grad=True)

        torch.nn.init.trunc_normal_(self.W,
                                    mean=0,
                                    std=1.,
                                    a=-1.,
                                    b=1.)
    def forward(self,tokens_ids):
    # Apply the embedding layer, cette function permet d'appliquer la fonction à chacun des tokens
        return self.W[tokens_ids]

class rmsnorm(nn.Module):
    """ 
    Apply RMSnorm
    params: epsilon: float, gain : float, d_model
    Normalize
    """
    def __init__(self, d_model, gain, epsilon=1e-5,device=None,dtype=None):
        super().__init__()
        self.d_model=d_model
        self.gain=nn.Parameter(torch.empty(d_model))
        self.epsilon=epsilon

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
        self.W_1=nn.Parameter(torch.empty(d_ff,d_model))
        self.W_2=nn.Parameter(torch.empty(d_model,d_ff))
        self.W_3=nn.Parameter(torch.empty(d_ff,d_model))
        self.d_model=d_model
        self.d_ff=d_ff
    def forward(self,x):
        if (self.d_ff-8/3*self.d_model)**2 > 1000*self.d_model:
            raise ValueError('d_ff is not approximataly 8/3 d_model')
        if (self.d_ff % 64)!=0:
            raise ValueError('d_ff is not a multiple of 64')
        w_1=einsum(x,self.W_1.T,'... d_model, d_model d_ff-> ... d_ff')
        SiLU=torch.sigmoid(w_1)* w_1
        w_3=einsum(x,self.W_3.T,'... d_model, d_model d_ff->... d_ff')
        z=w_3*SiLU
        x=einsum(z,self.W_2.T,'... d_ff, d_ff d_model->... d_model')
        return x


class RotaryPositionalEmbedding(nn.Module):
    """
    Apply RoPE 
    params: theta: float, d_k: int, max_seq_len: int
    """
    def __init__(self,theta,d_k,max_seq_len,device=None):
        super.__init__()
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
    scores=einsum(query,key,' b _ i q , b _ j q -> b _ i j')



