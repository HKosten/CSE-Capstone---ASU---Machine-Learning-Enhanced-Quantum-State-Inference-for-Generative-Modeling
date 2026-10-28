## How to Run and Modify the Tests

### Running the Tests

The `Main.py` file is located inside the `Team` folder. This file is used to run and evaluate the different models using the shared evaluation framework.

To run the tests:

1. Open the project folder in VS Code or another Python editor.
2. Make sure the required Python libraries are installed.
3. Open a terminal in the main project folder (where this README is located).
4. Run the following command:

`python Team/Main.py`

The program will run the enabled model tests and print their results in the terminal.

### Changing the Test Settings

The test settings can be changed directly in `Team/Main.py`.

The file contains a settings dictionary with the following values:

- `training_data`: The dataset being used for testing.
- `num_epochs`: The number of training epochs.
- `seed`: The random seed used during testing.
- `epsilon`: The proportion of the dataset used for training.
- `num_samples`: The number of samples the model should generate.
- `show_output`: Controls whether the generated samples are printed.

These values can be modified depending on what we want to test. For example, we can change the number of epochs or epsilon value to see how they affect the model's performance.

### Testing Different Datasets

The current `Main.py` includes tests for four datasets:

- DNA
- Qubit
- Sin
- Bars and Stripes

To change a dataset, update the `settings["training_data"]` path in the corresponding section of `Main.py`.

Each dataset has its own testing section, so changing one dataset path does not automatically change the others.

### Choosing Which Models to Test

The file includes tests for the LSTM, Transformer, VAE, and QCBM models.

To disable a test, add `#` at the beginning of its function call. To enable a commented-out test, remove the `#`.

Some tests are already commented out in the current file.

The QCBM also has a limit on the dataset size it can handle. If a dataset is too large, the program will skip that QCBM test and display a message instead.

### Understanding the Results

The evaluation framework uses Maximum Mean Discrepancy (MMD) to compare generated samples with the test data.

After each enabled test, the program prints the MMD score in the terminal. Lower MMD values generally indicate that the generated samples are closer to the test data distribution.

We can compare these scores across different datasets and settings to better understand how each model performs.

When comparing results, we should try to keep the other testing settings consistent so the comparison is fair.
