import evaluation_framework as ef
import LSTM
import transformer_wrapper as tw
import vae_wrapper as vw
import qcbm_wrapper as qw

def print_separators(model_name):
    
    if model_name == "LSTM" :
        print("\n ----------------------------------------------------- LSTM ------------------------------------------------------- \n")

    elif model_name == "Transformer":
        print("\n -------------------------------------------------- TRANSFORMER ---------------------------------------------------- \n")

    elif model_name == "VAE":
        print("\n ----------------------------------------------------- VAE -------------------------------------------------------- \n")

    elif model_name == "QCBM":
        print("\n ----------------------------------------------------- QCBM ------------------------------------------------------- \n")

def run_qcbm(settings):
    """
    the QCBM simulates 2^n numbers, so it only runs on data up to 16 bits (the 12-bit qubit set).
    wider data raises a ValueError before training starts; print it and keep going with the other models
    """
    try:
        ef.evaluate_model(qw.qcbm_function, settings)
    except ValueError as e:
        print(f"QCBM skipped: {e}")


settings = {"training_data": "training_sequences_DNA.csv", "num_epochs": 100, "seed": 33, "epsilon": 0.3, "num_samples": 100}

print_separators("LSTM")
ef.evaluate_model(LSTM.LSTM, settings)
print_separators("Transformer")
ef.evaluate_model(tw.transformer_function, settings)

print_separators("VAE")
ef.evaluate_model(vw.vae_function, settings)
print_separators("QCBM")
run_qcbm(settings)

settings["training_data"] = "training_sequences_qubit.csv"

print_separators("LSTM")
ef.evaluate_model(LSTM.LSTM, settings)
print_separators("Transformer")
ef.evaluate_model(tw.transformer_function, settings)
print_separators("VAE")
ef.evaluate_model(vw.vae_function, settings)
print_separators("QCBM")
run_qcbm(settings)

settings["training_data"] = "sin_bitstrings.csv"

print_separators("LSTM")
ef.evaluate_model(LSTM.LSTM, settings)
print_separators("Transformer")
ef.evaluate_model(tw.transformer_function, settings)
print_separators("VAE")
ef.evaluate_model(vw.vae_function, settings)
print_separators("QCBM")
run_qcbm(settings)

settings["training_data"] = "bars_stripes_bitstrings.csv"

print_separators("LSTM")
ef.evaluate_model(LSTM.LSTM, settings)
print_separators("Transformer")
ef.evaluate_model(tw.transformer_function, settings)
print_separators("VAE")
ef.evaluate_model(vw.vae_function, settings)
print_separators("QCBM")
run_qcbm(settings)
