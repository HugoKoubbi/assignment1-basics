def compute_parameters(num_layers, d_model, d_ff, vocab_size):
    part_QK_depth = num_layers * 4 * d_model**2 
    part_feedforward = num_layers * 3 * d_ff * d_model 
    embedding = vocab_size * d_model
    parameters = part_QK_depth + part_feedforward + embedding
    return parameters

def compute_non_embedding_parameters(num_layers, d_model, d_ff):
    part_QK_depth = num_layers * 4 * d_model**2 
    part_feedforward = num_layers * 3 * d_ff * d_model 
    parameters = part_QK_depth + part_feedforward
    return parameters