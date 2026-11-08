import evaluation_framework as ef
import LSTM
import transformer_wrapper as tw
import vae_wrapper as vw
import qcbm_wrapper as qw
import numpy as np

count = 1
def print_separators(model_name, settings):
    global count
    print(f"\n --------------------------------- Test #{count} | Model: {model_name} | dataset: {settings['training_data']} | epochs count: {settings['num_epochs']} | seed: {settings['seed']} | epsilon: {settings['epsilon']:.1f} | sample count: {settings['num_samples']} | show_output: {settings['show_output']} | ----------------------------------- \n")
    count += 1

def run_qcbm(settings, test_name):
    """
    the QCBM simulates 2^n numbers, so it only runs on data up to 16 bits (the 12-bit qubit set).
    wider data raises a ValueError before training starts; print it and keep going with the other models
    """
    try:
        ef.evaluate_model(qw.qcbm_function, settings, "QCBM", test_name)
    except ValueError as e:
        print(f"QCBM skipped: {e}")

# EPSILON TEST: varies epsilon from 0.1 to 0.9, incrementing by 0.1 each time
def epsilon_test(model, settings, model_name, test_name):
    
    test_settings = settings

    for i in np.arange(0.1, 1.0, 0.1):
        test_settings["epsilon"] = i
        print_separators(model_name, test_settings)
        ef.evaluate_model(model, test_settings, model_name, test_name)

# EPOCHS TEST: varies num_epochs from 0 to 10000, incrementing every 100 for now to test for flattening
def epochs_test(model, settings, model_name, test_name):
    
    test_settings = settings
    
    for i in range(0, 10000, 100):
        test_settings["num_epochs"] = i
        print_separators(model_name, test_settings)
        ef.evaluate_model(model, test_settings, model_name, test_name)


settings = {"training_data": "training_sequences_DNA.csv", "num_epochs": 100, "seed": 33, "epsilon": 0.3, "num_samples": 1000, "show_output": False}

# VARYING EPSILON:
# All other original settings remain the same except epsilon
# Every possible model passed in and tested. 
# Currently on the DNA dataset, change the filename in the settings dictionary and the curr_dataset_name to change
models = {"LSTM": LSTM.LSTM, "Transformer":tw.transformer_function, "VAE":vw.vae_function}
curr_dataset_name = "DNA"
"""
for model_name, model in models.items():
    epsilon_test(model, settings, model_name, curr_dataset_name)
"""
# VARYING NUM EPOCHS:
# All other original settings remain the same except num_epochs
for model_name, model in models.items():
    epochs_test(model, settings, model_name, curr_dataset_name)

# Potential new way to run tests?
for model_name, model in models.items():
    print_separators(model_name, settings)
    ef.evaluate_model(model, settings, model_name, curr_dataset_name)
run_qcbm(settings, curr_dataset_name)

# ORIGINAL TESTS
print_separators("LSTM", settings)
ef.evaluate_model(LSTM.LSTM, settings, "LSTM", "DNA")
print_separators("Transformer", settings)
ef.evaluate_model(tw.transformer_function, settings, "Transformer", "DNA")

print_separators("VAE", settings)
ef.evaluate_model(vw.vae_function, settings, "VAE", "DNA")
print_separators("QCBM", settings)
run_qcbm(settings, "DNA")

settings["training_data"] = "training_sequences_qubit.csv"

print_separators("LSTM", settings)
ef.evaluate_model(LSTM.LSTM, settings, "LSTM", "Qubit")
print_separators("Transformer", settings)
ef.evaluate_model(tw.transformer_function, settings, "Transformer", "Qubit")
print_separators("VAE", settings)
ef.evaluate_model(vw.vae_function, settings, "VAE", "Qubit")
print_separators("QCBM", settings)
run_qcbm(settings, "Qubit")

settings["training_data"] = "sin_bitstrings.csv"

print_separators("LSTM", settings)
ef.evaluate_model(LSTM.LSTM, settings, "LSTM", "Sin")
print_separators("Transformer", settings)
ef.evaluate_model(tw.transformer_function, settings, "Transformer", "Sin")
print_separators("VAE", settings)
ef.evaluate_model(vw.vae_function, settings, "VAE", "Sin")
print_separators("QCBM", settings)
run_qcbm(settings, "Sin")

settings["training_data"] = "bars_stripes_bitstrings.csv"

print_separators("LSTM", settings)
ef.evaluate_model(LSTM.LSTM, settings, "LSTM", "Bars+Stripes")
print_separators("Transformer", settings)
ef.evaluate_model(tw.transformer_function, settings, "Transformer", "Bars+Stripes")
print_separators("VAE", settings)
ef.evaluate_model(vw.vae_function, settings, "VAE", "Bars+Stripes")
print_separators("QCBM", settings)
run_qcbm(settings, "Bars+Stripes")
