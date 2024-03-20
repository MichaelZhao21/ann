# ANN (Artificial Neural Network) from scratch!

This is an Artificial Neural Network that performs a stochastic gradient descent.

## How to run

The main runner code is in `main.py` while the model code is placed in `model.py`.

To install libraries, create a [virtual environment](https://docs.python.org/3/library/venv.html) and install the required libraries with:

```sh
# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Install libraries
python3 -m pip install -r requirements.txt
# or
pip install -r requirements.txt
```

The configuration file (`config.txt`) stores the hyperparameters to test, in the following format:

```
<activation function> <learning rate> <iterations>
eg:           sigmoid             0.1          100 
// and so on
```

Run the model and tests with:

```sh
python3 main.py
```

You will be presented in your terminal with the epochs and accuracy at each epoch as well as a tabular form of the overall accruacies and parameters. The table will be rendered to `results.png`.

## Train Test Split

We chose a train/test split of 10% since there is not a whole lot of data to train with.

