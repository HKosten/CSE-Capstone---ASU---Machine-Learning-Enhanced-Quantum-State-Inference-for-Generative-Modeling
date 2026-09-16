# Model 1 Results — Autoregressive Coin Flip Model

## Training Results
- Model was trained over 500 epochs on qubit bitstring sequences
- Training loss decreased steadily throughout training, confirming the model successfully learned the patterns in the data
- Final training loss converged to a very low value, demonstrating stable and accurate learning

## Probability Distribution Results
- Model successfully outputs a complete probability distribution over all possible qubit bitstring outcomes
- The full probability distribution sums to **1.000000**, confirming mathematical validity
- Top outcomes are assigned equal learned probabilities based on patterns in the training data
- Outcomes seen in training data are assigned higher learned probabilities
- Unseen/new outcomes (e.g., All Heads) are assigned a prior probability, ensuring no outcome is excluded from the distribution

## Prediction Accuracy
- Model correctly predicted outcomes across all tested sequences
- Sequences with a history of mixed observations correctly output lower probability of Heads
- Sequences with a history of all Tails correctly output higher probability of Heads
- Predicted outcomes matched actual outcomes for every test case evaluated

## Comparison to True Quantum Distribution
- MMD (Maximum Mean Discrepancy) evaluation against the true quantum distribution is currently in progress
- MMD will be used to quantitatively compare Model 1 against Model 2 (LSTM) and Model 3 (Encoder-Decoder Transformer)

## Preliminary Conclusion
The Autoregressive Coin Flip Model demonstrates that a classical sequential neural network can successfully learn and output a valid probability distribution over qubit bitstring outcomes. The model handles both observed and unseen sequences, making it robust to sparse training data.
