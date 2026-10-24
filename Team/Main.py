import evaluation_framework as ef
import LSTM
import transformer_wrapper as tw
import vae_wrapper as vw
import qcbm_wrapper as qw

def print_separators(model_name, test_name):
    
    print(f"\n ------------------------------------------------ Model: {model_name}, Test: {test_name} -------------------------------------------------- \n")

def run_qcbm(settings):
    """
    the QCBM simulates 2^n numbers, so it only runs on data up to 16 bits (the 12-bit qubit set).
    wider data raises a ValueError before training starts; print it and keep going with the other models
    """
    try:
        ef.evaluate_model(qw.qcbm_function, settings)
    except ValueError as e:
        print(f"QCBM skipped: {e}")


settings = {"training_data": "Team\\training_sequences_DNA.csv", "num_epochs": 100, "seed": 33, "epsilon": 0.3, "num_samples": 1000, "show_output": False}

# print_separators("LSTM", "DNA")
# ef.evaluate_model(LSTM.LSTM, settings)
# print_separators("Transformer", "DNA")
# ef.evaluate_model(tw.transformer_function, settings)

# print_separators("VAE", "DNA")
# ef.evaluate_model(vw.vae_function, settings)
print_separators("QCBM", "DNA")
run_qcbm(settings)

settings["training_data"] = "Team\\training_sequences_qubit.csv"

print_separators("LSTM", "Qubit")
ef.evaluate_model(LSTM.LSTM, settings)
print_separators("Transformer", "Qubit")
ef.evaluate_model(tw.transformer_function, settings)
print_separators("VAE", "Qubit")
ef.evaluate_model(vw.vae_function, settings)
print_separators("QCBM", "Qubit")
run_qcbm(settings)

settings["training_data"] = "Team\\sin_bitstrings.csv"

print_separators("LSTM", "Sin")
ef.evaluate_model(LSTM.LSTM, settings)
print_separators("Transformer", "Sin")
ef.evaluate_model(tw.transformer_function, settings)
print_separators("VAE", "Sin")
ef.evaluate_model(vw.vae_function, settings)
print_separators("QCBM", "Sin")
run_qcbm(settings)

settings["training_data"] = "Team\\bars_stripes_bitstrings.csv"

print_separators("LSTM", "Bars+Stripes")
ef.evaluate_model(LSTM.LSTM, settings)
print_separators("Transformer", "Bars+Stripes")
ef.evaluate_model(tw.transformer_function, settings)
print_separators("VAE", "Bars+Stripes")
ef.evaluate_model(vw.vae_function, settings)
print_separators("QCBM", "Bars+Stripes")
run_qcbm(settings)
