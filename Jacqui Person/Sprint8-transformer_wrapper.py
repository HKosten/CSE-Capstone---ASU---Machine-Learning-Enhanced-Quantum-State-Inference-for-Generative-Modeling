from transformer import transformer 
import torch
import pandas as pd

def transformer_function(trainingData):

    bitstring_df = pd.read_csv(trainingData, dtype={"bitstring": str})
    data = bitstring_df["bitstring"].apply(list)
    data = data.apply(lambda x: [int(bit) for bit in x])
    input_data = torch.tensor(data, dtype=torch.long)
    
    transformer.eval() 

    with torch.no_grad():

        outputs = transformer(input_data, input_data) 
        probs = torch.softmax(outputs, dim=-1)

        generated_samples = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(probs.size(0), probs.size(1))

    return generated_samples
