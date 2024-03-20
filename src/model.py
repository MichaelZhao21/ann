from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import numpy as np


class Model:
    """
    Simple ANN with backpropagation

    Attributes
    ----------
    layers : list
        List of integers representing the number of nodes in each layer
    activation : str
        String representing the activation function to use (sigmoid, tanh, relu)
    split_percent : float
        Percentage of data to use for testing
    learn_rate : float
        Learning rate for backpropagation
    iterations : int
        Number of iterations to run backpropagation with (loop through all training examples this many times)
    inputs : list
        List of lists, each representing the input nodes for each node
    outputs : list
        List of lists, each representing the output nodes for each node
    weights : np.array
        Matrix of weights, where (r, c) represents the weight from node r to node c
    biases : list
        List of biases for each node
    n : int
        Total number of nodes in the network
    X_train : np.array
        Training inputs
    X_test : np.array
        Testing inputs
    y_train : np.array
        Training outputs
    y_test : np.array
        Testing outputs
    act_func : function
        Activation function
    d_act_func : function
        Derivative of the activation function
    train_size : list
        List of the number of correct predictions and the total number of training examples
    train_acc : float
        Training accuracy
    test_size : list
        List of the number of correct predictions and the total number of testing examples
    test_acc : float
        Testing accuracy


    Methods
    -------
    pre_process(df: pd.DataFrame) -> (X, y)
        Preprocesses the dataframe, one hot encoding the outputs and scaling the inputs
    train()
        Trains the model with the training data
    test()
        Tests the model with the testing data
    """

    def __init__(
        self,
        layers,
        activation="sigmoid",
        split_percent=0.1,
        learn_rate=0.1,
        iterations=100,
    ):
        """
        Parameters
        ----------
        layers : list
            List of integers representing the number of nodes in each layer
        activation : str
            String representing the activation function to use (sigmoid, tanh, relu)
        split_percent : float
            Percentage of data to use for testing
        learn_rate : float
            Learning rate for backpropagation
        iterations : int
            Number of iterations to run backpropagation with (loop through all training examples this many times)
        """
        # Define the model representation
        self.layers = layers
        self.inputs = []
        self.outputs = []
        self.weights = np.zeros((sum(self.layers), sum(self.layers)))
        self.biases = []
        self.n = 0  # Current num of nodes created

        ##### CREATE THE NETWORK (DEFINE NODES, EDGES, RAND WEIGHTS) #####

        # Loop through all layers, where layer_nodes = nodes in curr layer
        for layer_idx, layer_nodes in enumerate(self.layers):

            # Create inputs and outputs for the current layer (loop through all nodes in the layer)
            for _ in range(self.n, self.n + layer_nodes):

                # If we are in an input layer, no input connections
                # Otherwise, add all previous layer's nodes
                if layer_idx == 0:
                    self.inputs.append([])
                else:
                    self.inputs.append(
                        list(range(self.n - self.layers[layer_idx - 1], self.n))
                    )

                # If we are in an output layer, no output connections
                # Otherwise, add all next layer's nodes
                if layer_idx == len(self.layers) - 1:
                    self.outputs.append([])
                else:
                    self.outputs.append(
                        list(
                            range(
                                self.n + layer_nodes,
                                self.n + layer_nodes + self.layers[layer_idx + 1],
                            )
                        )
                    )

                # Create a bias for each node (no bias for input nodes)
                if layer_idx == 0:
                    self.biases.append(0.0)
                else:
                    self.biases.append(np.random.normal(scale=0.25))

            # Increment total num of nodes
            self.n += layer_nodes

        # Create an upper triangular matrix of weights
        for node, row in enumerate(self.inputs):
            for cell in row:
                self.weights[cell][node] = np.random.normal(scale=0.25)

        # Activation functions
        sigmoid = lambda x: 1 / (1 + math.e ** (-x))
        tanh = lambda x: (math.e ** (x) - math.e ** (-x)) / (
            math.e ** (x) + math.e ** (-x)
        )
        relu = lambda x: x if x > 0 else 0

        # Derivatives of the activation functions
        d_sigmoid = lambda x: x * (1 - x)
        d_tanh = lambda x: 1 - x**2
        d_relu = lambda x: 1 if x > 0 else 0

        # Make sure activation function is valid
        valid_funcs = ["sigmoid", "tanh", "relu"]
        if activation not in valid_funcs:
            raise Exception(
                f'{activation} not a valid activation function, must be one of [{", ".join(valid_funcs)}]'
            )

        # Assign activation function
        activations = [sigmoid, tanh, relu]
        d_activations = [d_sigmoid, d_tanh, d_relu]
        self.act_func = activations[valid_funcs.index(activation)]
        self.d_act_func = d_activations[valid_funcs.index(activation)]

        # Assign other parameters to attributes
        self.learn_rate = learn_rate
        self.iterations = iterations
        self.split_percent = split_percent
        self.activation = activation

    def pre_process(self, df: pd.DataFrame):
        """
        Preprocesses the data, one hot encoding the outputs and scaling the inputs

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the data
        """
        # Create one-hot encoded classes for the outputs
        unique_classes = list(set(df["class"]))
        df["class_encoded"] = df["class"].apply(
            lambda x: [(1 if x == c else 0) for c in unique_classes]
        )

        # Extract X and y (inputs and outputs)
        X, y = df.iloc[:, 0:4], df.iloc[:, 5]

        # Scale the data
        scaler = StandardScaler()
        X = scaler.fit(X).transform(X)

        self.X_train, self.X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.split_percent
        )
        # Reset index after splitting
        self.y_train = y_train.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)

    def train(self):
        """
        Trains the model with the training data
        """
        # Array for outputs (os), delta of each node (deltas),
        # and total number of output nodes
        os = [0] * self.n
        deltas = [0] * self.n
        output_nodes = self.layers[-1]
        y_comp = [x for y in self.y_train for x in y]  # Flattened output array

        # Loop for defined iterations
        for i in range(self.iterations):

            # Define variables for accuracy and error calculation
            correct = 0
            all_test_os = []

            # Loop through all training examples
            for j, input in enumerate(self.X_train):

                # 1. Compute the outputs of the network
                # Copy over the input values to the input layer
                for k, val in enumerate(input):
                    os[k] = val

                # Evaluate the outputs of each hidden and output node, in node order
                for k in range(len(input), self.n):
                    curr_weights = [self.weights[x][k] for x in self.inputs[k]]
                    summation = [
                        x[0] * x[1]
                        for x in zip(curr_weights, [os[y] for y in self.inputs[k]])
                    ]
                    os[k] = self.act_func(sum(summation) + self.biases[k])

                # 2. Calculate the delta_k for output units
                for k in range(output_nodes):
                    idx = self.n - output_nodes + k  # Index of outputs/deltas array
                    deltas[idx] = self.d_act_func(os[idx]) * (
                        self.y_train[j][k] - os[idx]
                    )

                # 3. Calculate the delta_h for hidden units
                for k in range(self.layers[0], self.n - output_nodes):
                    curr_outputs = self.outputs[k]
                    curr_ws = [self.weights[k][x] for x in curr_outputs]
                    curr_deltas = [deltas[o] for o in curr_outputs]
                    deltas[k] = self.d_act_func(os[k]) * sum(
                        [x[0] * x[1] for x in zip(curr_ws, curr_deltas)]
                    )

                # 4. Update each weight
                for k, row in enumerate(self.inputs):
                    for cell in row:
                        self.weights[cell][k] += self.learn_rate * deltas[k] * os[cell]

                # 4b. Update biases
                for k in range(self.layers[0], self.n):
                    self.biases[k] += self.learn_rate * deltas[k]

                # Add final output to all test output array
                all_test_os.extend(os[-3:])
                if self.max_index(self.y_train[j]) == self.max_index(os[-3:]):
                    correct += 1

            # Perform the error calculation for the current iteration
            err = (1 / (2 * len(y_comp))) * sum(
                [(a[0] - a[1]) ** 2 for a in zip(y_comp, all_test_os)]
            )
            print(f"Iteration {i+1}: {err} | {correct}/{len(self.y_train)}")

        # Store training size and accuracy of last iteration for output
        self.train_size = [correct, len(self.y_train)]
        self.train_acc = 100 * (correct / len(self.y_train))  # train accuracy

    def test(self):
        """
        Tests the model with the testing data
        """
        correct = 0
        os = [0] * self.n
        check = []
        y_comp = [x for y in self.y_test for x in y]

        # Loop through all testing examples
        for j, input in enumerate(self.X_test):

            # Copy over the input values to the input layer
            for k, val in enumerate(input):
                os[k] = val

            # Evaluate the outputs of each hidden and output node, in node order
            for k in range(len(input), self.n):
                curr_weights = [self.weights[x][k] for x in self.inputs[k]]
                summation = [
                    x[0] * x[1]
                    for x in zip(curr_weights, [os[y] for y in self.inputs[k]])
                ]
                os[k] = self.act_func(sum(summation) + self.biases[k])

            # Add final output to all test output array
            check.extend(os[-3:])

            # Increment correct if the output is correct
            if self.max_index(self.y_test[j]) == self.max_index(os[-3:]):
                correct += 1

        # Perform the error calculation for the testing data
        err = (1 / (2 * len(y_comp))) * sum(
            [(a[0] - a[1]) ** 2 for a in zip(y_comp, check)]
        )

        # Output and store the testing accuracy
        print(
            f"Test Accuracy for {len(self.y_test)} examples: {100*(correct/len(self.y_test))}% | Passed: {correct}/{len(self.y_test)}"
        )
        print(f"Test Error: {err}")
        self.test_size = [correct, len(self.y_test)]
        self.test_acc = 100 * (correct / len(self.y_test))

    def max_index(self, arr):
        """
        Returns the index of the maximum value in the array (helper function)
        """
        max_val = max(arr)
        return arr.index(max_val)
