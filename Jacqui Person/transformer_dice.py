import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import random
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ignite.metrics import MaximumMeanDiscrepancy # type: ignore

def calculate_mmd(generated_samples, real_samples):

    mmd = MaximumMeanDiscrepancy()
    mmd.update((generated_samples.float(), real_samples.float()))

    print("MMD: ", mmd.compute())

    return mmd.compute()

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


#PREPARE DATA FOR TRAINING !

#Load in new bitstring dataset
#1 sample = 1 bitstring
#Split each string into bits & convert bits to integers
bitstring_df = pd.read_csv("bitstrings.csv", dtype={"bitstring": str})
data = bitstring_df["bitstring"].apply(list)
data = data.apply(lambda x: [int(bit) for bit in x])
data = torch.tensor(data, dtype=torch.long)

#Split data - training and eval 80/20
train_data, test_data = train_test_split(data, train_size = 0.80, random_state=50)

#Print shapes (rows, columns)
print("TRAINING DATA: ", train_data.shape)
print("TESTING DATA: ", test_data.shape)

#Remember than MMD EVAL will use test_data, not train_data!
src_data = train_data
tgt_data = train_data

#TRAINING !

#CHANGED from qubit to seq_length: length of each bitstring
#vocab_size is size for binary strings
src_vocab_size = 2
tgt_vocab_size = 2
seq_length = 128

#Decreased d_model 512->64, num_heads 8->4, num_layers 4->2
d_model = 64
num_heads = 4
num_layers = 2
d_ff = 128
max_seq_length = seq_length
dropout = 0.1

#Transformer instance
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

#optimizer = Adam
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# USE CROSS ENTROPY LOSS DURING TRAINING INSTEAD OF MMD
criterion = nn.CrossEntropyLoss()

#Model is in training mode
transformer.train()

#Train over 100 epochs - maybe more here!
#!!! Max. mean discrepancy - how close two distributions are using just samples, don't need entire distribution
for epoch in range(100):
    optimizer.zero_grad()

    output = transformer(src_data, tgt_data)
    probs = torch.softmax(output, dim=-1)

    loss = criterion(output.view(-1, tgt_vocab_size), tgt_data.view(-1))

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

with torch.no_grad():
    outputs = transformer(src_data, tgt_data)
    probs = torch.softmax(outputs, dim=-1)

    #Sample from the distribution and flatten
    model_samples = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(probs.size(0), probs.size(1))

#Performance evaluation
transformer.eval()

#Disables gradient computation (no need during validation)
#MMD,
with torch.no_grad():
   
    real_samples = test_data.float()
    generated_samples = model_samples.float()

#Trim both sets of samples since MMD needs datasets to be comparable in size & alignment
    min_samples = min(model_samples.size(0), real_samples.size(0))
    generated_samples = generated_samples[:min_samples]
    real_samples = real_samples[:min_samples]

    calculate_mmd(generated_samples=generated_samples, real_samples=real_samples)




