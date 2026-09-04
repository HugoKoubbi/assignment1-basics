#### GPT2-small

vocab_size= 50257
context_length= 1024
num_layers= 12
d_model=768
num_heads=25
d_ff=8*d_model/3 

part_QK_depth =96* 4 * d_model**2 * context_length
part_scores = 4* 48 * context_length**2 *d_ff
part_feedforward = 48*3*2*d_ff* d_model*context_length  
embedding=vocab_size*d_model*context_length*2
total=part_QK_depth+part_scores+part_feedforward+embedding

print(f'GPT2-XL small -> % QK_depth:{part_QK_depth/total}')
print(f'GPT2-XL small -> % scores_depth:{part_scores/total}')
print(f'GPT2-XL small -> % scores_depth:{part_feedforward/total}')
print(f'GPT2-XL small -> % embedding:{embedding/total}')
print(f'GPT2 small-> Flops:{total}')


#### GPT2-medium

vocab_size= 50257
context_length= 1024
num_layers= 24
d_model=1024
num_heads=25
d_ff= 8*d_model/3 

part_QK_depth =96* 4 * d_model**2 * context_length
part_scores = 4* 48 * context_length**2 *d_ff
part_feedforward = 48*3*2*d_ff* d_model*context_length  
embedding=vocab_size*d_model*context_length*2
total=part_QK_depth+part_scores+part_feedforward+embedding

print(f'GPT2-XL medium -> % QK_depth:{part_QK_depth/total}')
print(f'GPT2-XL medium -> % scores_depth:{part_scores/total}')
print(f'GPT2-XL medium -> % scores_depth:{part_feedforward/total}')
print(f'GPT2-XL medium -> % embedding:{embedding/total}')
print(f'GPT2 medium -> Flops:{total}')


### GPT2 large
#### GPT2-medium

vocab_size= 50257
context_length= 1024
num_layers= 36
d_model=1280
num_heads=25
d_ff= 8*d_model/3 

part_QK_depth =96* 4 * d_model**2 * context_length
part_scores = 4* 48 * context_length**2 *d_ff
part_feedforward = 48*3*2*d_ff* d_model*context_length  
embedding=vocab_size*d_model*context_length*2
total=part_QK_depth+part_scores+part_feedforward+embedding

print(f'GPT2 large -> % QK_depth:{part_QK_depth/total}')
print(f'GPT2 large -> % scores_depth:{part_scores/total}')
print(f'GPT2 large -> % scores_depth:{part_feedforward/total}')
print(f'GPT2 large -> Flops:{total}')



#### GPT2-xlarge

vocab_size= 50257
context_length= 1024
num_layers= 48
d_model=1600
num_heads=25
d_ff=4288 


part_QK_depth =96* 4 * d_model**2 * context_length
part_scores = 4* 48 * context_length**2 *d_ff
part_feedforward = 48*3*2*d_ff* d_model*context_length  
embedding=vocab_size*d_model*context_length*2
total=part_QK_depth+part_scores+part_feedforward+embedding

print(f'GPT2-XL large -> % QK_depth:{part_QK_depth/total}')
print(f'GPT2-XL large -> % scores_depth:{part_scores/total}')
print(f'GPT2-XL large -> % scores_ffn:{part_feedforward/total}')
print(f'GPT2-XL large -> % embedding:{embedding/total}')
print(f'GPT2-XL large -> Flops:{total}')


#### With new context length

#### GPT2-xlarge

vocab_size= 50257
context_length= 16384
num_layers= 48
d_model=1600
num_heads=25
d_ff=4288 


part_QK_depth =96* 4 * d_model**2 * context_length
part_scores = 4* 48 * context_length**2 *d_ff
part_feedforward = 48*3*2*d_ff* d_model*context_length  
embedding=vocab_size*d_model*context_length*2
total=part_QK_depth+part_scores+part_feedforward+embedding

print(f'GPT2-XL large with nw context-> % QK_depth:{part_QK_depth/total}')
print(f'GPT2-XL large with nw context-> % scores_depth:{part_scores/total}')
print(f'GPT2-XL large with nw context-> % scores_ffn:{part_feedforward/total}')
print(f'GPT2-XL large with nw context -> % embedding:{embedding/total}')
print(f'GPT2-XL large with nw context -> Flops:{total}')


### AdamW memory
vocab_size= 50257
n= 1024
num_layers= 48
d=1600
h=25
dff=4288 
b=6

activations=num_layers * b * h * n**2 + (num_layers+2)* 6* b * n * d + num_layers * 4 * b * n * dff + b * n * vocab_size

parameters_2=0
part_QK_depth_2 = num_layers * 4 * d**2 
part_feedforward_2 = num_layers * 3 * dff * d 
embedding_2 = vocab_size * d
parameters_2 = part_QK_depth_2+part_feedforward_2+embedding_2

state=2*parameters_2
gradient=parameters_2

total=state+gradient+parameters_2+activations
total_bytes = 4*total
total_Gb= 4* total * 10**(-9)
parameters_Gb=4* parameters_2 * 10**(-9)
activations_Gb=4* activations * 10**(-9)

print(f'The total memory (in GB) for b={b} -> {total_Gb}')
print(f'The total memory (in Gb) for parameters -> {parameters_Gb}')
print(f'The total memory (in Gb) for activation -> {activations_Gb}')


