import evaluation_framework as ef
import LSTM
import transformer_wrapper as tw

def print_separators(model_name):
    
    if model_name == "LSTM" :
        print("\n ----------------------------------------------------- LSTM ------------------------------------------------------- \n")

    elif model_name == "Transformer":
        print("\n -------------------------------------------------- TRANSFORMER ---------------------------------------------------- \n")


settings = {"training_data": "training_sequences_DNA.csv", "num_epochs": 100, "seed": 33, "epsilon": 0.3, "num_samples": 100}

print_separators("LSTM")
ef.evaluate_model(LSTM.LSTM, settings)
print_separators("Transformer")
ef.evaluate_model(tw.transformer_function, settings)

# ef.evaluate_model(<faisal function here>, settings) 
# ef.evaluate_model(<Khashim function here>, settings) 

settings["training_data"] = "training_sequences_qubit.csv"

print_separators("LSTM")
ef.evaluate_model(LSTM.LSTM, settings)
print_separators("Transformer")
ef.evaluate_model(tw.transformer_function, settings)
# ef.evaluate_model(<faisal function here>, settings) 
# ef.evaluate_model(<Khashim function here>, settings) 

settings["training_data"] = "sin_bitstrings.csv"

print_separators("LSTM")
ef.evaluate_model(LSTM.LSTM, settings)
print_separators("Transformer")
ef.evaluate_model(tw.transformer_function, settings)
# ef.evaluate_model(<faisal function here>, settings) 
# ef.evaluate_model(<Khashim function here>, settings) 
