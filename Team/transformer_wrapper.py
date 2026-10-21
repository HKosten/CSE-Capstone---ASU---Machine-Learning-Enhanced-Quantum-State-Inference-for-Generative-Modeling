from transformer import Transformer
from transformer import main_training_logic
import torch
import pandas as pd


def transformer_function(**settings):

    input_data = torch.as_tensor(settings["training_data"], dtype=torch.long)

    transformer = main_training_logic(training_data=input_data, num_epochs=settings["num_epochs"], seed=settings["seed"], epsilon=settings["epsilon"])

    transformer.train()

    num_samples = settings["num_samples"]

    indices = torch.randint(0, len(input_data), (num_samples,))
    sample_input = input_data[indices]
    
    transformer.eval() 

    with torch.no_grad():

        outputs = transformer(sample_input, sample_input) 
        probs = torch.softmax(outputs, dim=-1)

        generated_samples = torch.multinomial(probs.reshape(-1, probs.size(-1)), num_samples=1).reshape(probs.size(0), probs.size(1))

    return generated_samples


    