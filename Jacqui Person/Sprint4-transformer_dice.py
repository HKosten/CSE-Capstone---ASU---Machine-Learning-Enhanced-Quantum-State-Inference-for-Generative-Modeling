import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import random
import copy

#Dont print scientific notations, 4 dec. places
torch.set_printoptions(sci_mode=False, precision=4)

#Main idea - decoder generates the predicted sequence by using its own previous outputs as inputs to predict next token. 
# it "attends" the encoder's output.
#Encoder - takes input sequence & produces continuous, context-aware representation by using self-attn. 
# understands relationship between each word to every other word in input sequence

#"Attend" - dynamically calculate which other tokens in sequence are most relevant to the current token!
#Computes attention between each pair of positions in a sequence using mult. "attention heads"
#that capture diff. aspects of input seq.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output

#FFNN applied @ each position separately and identically
#to transform features learned by attn. mechs w/in transformer
#(addtl. processing step)    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

#Positional encoding - encode position of each token in input seq.  
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model) #A tensor filled with zeros, which will be populated with positional encodings.
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) #A tensor containing the position indices for each position in the sequence.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)) #A term used to scale the position indices in a specific way.
        
        pe[:, 0::2] = torch.sin(position * div_term) #Even indices of pe
        pe[:, 1::2] = torch.cos(position * div_term) #Odd indices of pe
        
        self.register_buffer('pe', pe.unsqueeze(0)) #register pe as buffer (part of mod. state, not trainable param.)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

#uses dropout, randomly "drop" neurons turing training to reg./prevent overfitting
#Capture complex relationships in input, transform into useful rep. for downstream tasks
#steps: 
# 1. self-attn. through multi-head self-attn. mech
# 2. Add and normalize - attn. output added to original input x, dropout, normalize (norm1)
# 3. FFNN - output from 2 passed thru pos.-wise FFNN
# 4. Add and normalize - FFNN output added to input of this stage, dropout, normalize (norm2)
# 5. Tensor is returned as output of encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout): #d_ff: dimensionality of inner layer in pos.-wise FFNN, dropout = dropout rate used for reg.
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
#Steps:
# 1. Input, X, processed thru self-attn. mech
# 2. Add and normalize - self-attn. output added to orig. X, dropout, normalization (norm1)
# 3. Cross-attn. - normalized output from step 2 processed thru cross attn. mech that attends encoder's output
# 4. Add and normalize - cross attn. output added to input of this stage, dropout, normalize (norm2)
# 5. FFNN - output from step 4 fed thru FFNN
# 6. Add and normalize - FFNN output added to input of this stage, dropout, normalization (norm3)
# 7. output = processed tensor
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads) #Multi-head self-attention mechanism for the target sequence.
        self.cross_attn = MultiHeadAttention(d_model, num_heads) #Multi-head attention mechanism that attends to the encoder's output.
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff) #Position-wise feed-forward neural network.
        self.norm1 = nn.LayerNorm(d_model) #Layer normalization components.
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) #Dropout layer for regularization.
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

#Steps:
# 1. source & target seq. embedded using respective embedding layers, added to positional encodings
# 2. source seq. passed thru encoder layers, final encoder ooutput represents processed source sequence
# 3. target seq. AND encoder's output passed thru decoder layers -> decoder output
# 4. decoder output mapped to target vocab. size using linear layer  
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model) #Embedding layer for the source sequence.
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model) #Embedding layer for the target sequence.
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length) #Positional encoding component.

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]) #A list of encoder layers
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]) #A list of decoder layers.

        self.fc = nn.Linear(d_model, tgt_vocab_size) #Final fully connected (linear) layer mapping to target vocabulary size.
        self.dropout = nn.Dropout(dropout) #Dropout layer.

# create masks for source and target sequences so padding tokens ignored, future tokens not visible during training for target seq.
# ensures predicted sequence is only based on its own prev. outputs
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
    

#TRAINING !

#update vocab size for dice
#Padded to 7 because need one token to serve at ground truth inside src vocab - aligns with the next prediction
#Feed correct previous token at each step to predict target
src_vocab_size = 7
tgt_vocab_size = 7

d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data
#src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
#tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

#new dice data - dice sequences with actual patterns for transformer to learn
#Tensor has this shape because feed transformer mult. sequences at once.
def generate_dependent_sequence(batch_size, seq_len):
    data = torch.zeros((batch_size, seq_len), dtype=torch.long) #Empty tensor of shape batch_size, seq_length to store sequences (each row = one sequence)

    #For each sequence in batch, randomly pick first roll.
    #Then, generate sequence:
    #If first roll = 3, P(1 | 3) = 0.1, P(2 | 3) = 0.1, etc.
    #If first roll = 6, P(1| 6) = 0.3...
    #Else, next rolls will just have a uniform distribution
    for b in range(batch_size):
        first = torch.randint(1, 7, (1,)).item()
        data[b, 0] = first

        for i in range(1, seq_len):
            if first == 3:
                probs = torch.tensor([0.1, 0.1, 0.3, 0.1, 0.1, 0.3])
            elif first == 6:
                probs = torch.tensor([0.3, 0.1, 0.1, 0.3, 0.1, 0.1])
            else:
                probs = torch.ones(6)/6

            next_roll = torch.multinomial(probs, 1).item() + 1 #sample index from created prob. dist - convert to dice val
            data[b, i] = next_roll #Append to current sequence
            first = next_roll #update prev. first roll value

    return data

# Generate dice rolls instead! 
data = generate_dependent_sequence(64, max_seq_length)
src_data = data
tgt_data = data

#Transformer instance
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

#loss function = Cross entropy loss
#optimizer = Adam
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

#Model is in training mode
transformer.train()

#Train over 100 epochs - maybe more here!
#!!! Max. mean discrepancy - how close two distributions are using just samples, don't need entire distribution


for epoch in range(100):
    optimizer.zero_grad() #Clears the gradients from the previous iteration.
    output = transformer(src_data, tgt_data[:, :-1]) #Pass src. data and target data thru transformer
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1)) #Computes the loss between the model's predictions and the target data
    loss.backward() #Backprop
    optimizer.step() #Update model's params using computed gradients
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

#Output probs for each next possible token, given the FIRST sequence in the batch at the FIRST time step
#Each probs[b, i] is full prob. dist. conditioned on src sequence, prev. target token
with torch.no_grad():
    outputs = transformer(src_data, tgt_data[:, :-1])
    probs = torch.softmax(outputs, dim=-1) 

    print("Probabilities of NEXT token at 0:", probs[0, 0])

    pred_seq = torch.argmax(probs, dim=-1)
    print("Predicted sequence:", pred_seq[0])

#Performance evaluation
transformer.eval()

# Generate dice sequence validation data!
val_data = generate_dependent_sequence(64, max_seq_length) #Batch size, seq length
val_src_data = val_data
val_tgt_data = val_data

#Disables gradient computation (no need during validation)
with torch.no_grad():

    val_output = transformer(val_src_data, val_tgt_data[:, :-1]) #Pass val. src. data and val. target data thru transformer
    val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1)) #Compute loss b/w model's predictions and val. target data
    
    val_probs = torch.softmax(val_output, dim=-1)
    print(f"Validation Loss: {val_loss.item()}")
    print("Predicted distribution:", val_probs[0, 0]) #Print predicted distribution of next token in a NEW sequence (unseen validation data)

