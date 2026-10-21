import evaluation_framework as ef
import LSTM
import transformer_wrapper as tw

settings = {"training_data": "training_sequences_DNA.csv", "num_epochs": 100, "seed": 33, "epsilon": 0.3, "num_samples": 100}

ef.evaluate_model(LSTM.LSTM, settings)
ef.evaluate_model(tw.transformer_function, settings)
# ef.evaluate_model(<faisal function here>, settings) 
# ef.evaluate_model(<Khashim function here>, settings) 

settings["training_data"] = "training_sequences_qubit.csv"

ef.evaluate_model(LSTM.LSTM, settings)
ef.evaluate_model(tw.transformer_function, settings)
# ef.evaluate_model(<faisal function here>, settings) 
# ef.evaluate_model(<Khashim function here>, settings) 