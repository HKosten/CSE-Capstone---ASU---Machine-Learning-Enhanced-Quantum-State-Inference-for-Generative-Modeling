# Faisal Al-Qahtani — Model 1: Autoregressive Coin Flip Model

## Project Overview
This folder contains my contributions to the CSE 485 Capstone Project:
**Machine Learning-Enhanced Quantum State Inference for Generative Modeling**, sponsored by Gennaro De Luca at Arizona State University.

The core problem this project addresses is that quantum computers cannot have their probability distributions directly extracted. Instead, the distribution must be estimated by repeatedly measuring the system, which becomes exponentially expensive as the number of qubits grows. My model uses a classical machine learning approach to estimate that distribution instead.

---

## My Model: Autoregressive Coin Flip Model

### What it does
The Autoregressive Coin Flip Model is a sequential neural network that predicts the probability of the next qubit outcome given a history of previous observations. Rather than simply generating samples, it outputs a **complete probability distribution** over all possible qubit bitstring outcomes.

### Model Architecture
- **Layer 0:** Linear (in_features=4, out_features=16)
- **Layer 1:** ReLU activation
- **Layer 2:** Linear (in_features=16, out_features=1)
- **Layer 3:** Sigmoid activation (outputs valid probability between 0 and 1)

### Key Features
- Outputs a full probability distribution that always sums to 1.0
- Observed sequences are assigned learned probabilities
- Unseen sequences are assigned a prior probability, ensuring complete coverage of the output space
- Trained over 500 epochs with steadily decreasing loss

---

## How to Run the Code

1. Open `Coin Flip Code` in Google Colab or Jupyter Notebook
2. Run all cells from top to bottom
3. The model will train and print:
   - Training loss at each epoch interval
   - Predictions vs actual outcomes for test sequences
   - Full probability distribution over all possible outcomes
   - Probability distribution sum (should equal 1.000000)

---

## Output Explanation
- **Training loss:** Shows how well the model is learning — lower is better
- **Predictions:** Shows model's predicted outcome vs the actual outcome for each test sequence
- **Probability distribution:** Lists every possible bitstring outcome and its assigned probability
- **Distribution sum:** Confirms mathematical validity — must equal 1.0
- **Unseen outcomes:** Any sequence not in the training data receives a prior probability

---

## Files in this Folder
| File | Description |
|------|-------------|
| `Coin Flip Code` | Main Autoregressive Coin Flip Model notebook |
| `Simple Coin Flip` | Earlier simpler version of the coin flip model |
| `Updates Regarding code` | Notes and updates on code changes |
| `MMD_Research.md` | Research notes on Maximum Mean Discrepancy implementation |
| `Results.md` | Preliminary results from Model 1 |
| `weather_ar.py` | Early autoregressive experiment using weather data |

---

## Next Steps
- Implement Maximum Mean Discrepancy (MMD) from scratch to evaluate model accuracy
- Compare Model 1 results against Model 2 (LSTM) and Model 3 (Encoder-Decoder Transformer)
- Investigate scalability of the autoregressive approach as the number of qubits increases
- Apply model to quantum circuit data provided by the project sponsor
