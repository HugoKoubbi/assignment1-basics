from cs336_basics.model import *

model=multihead_self_attention(d_model=12,num_heads=2,max_seq_len=10,rope_theta=10)
#x=torch.randn(torch.ones((1,10,12)))
#print(model(x))

x=torch.ones((1,10,10,12))
rope=RotaryPositionalEmbedding_gpt(d_k=12,max_seq_len=10,theta=10)
print(rope(x,torch.arange(10)))


