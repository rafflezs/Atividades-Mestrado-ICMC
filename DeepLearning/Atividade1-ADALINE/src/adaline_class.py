import numpy as np

np.random.seed(13)

class Adaline:
    """
    Adaline (Adaptive Linear Neuron) is a type of single-layer artificial neural network
    that can be used for binary classification tasks. It adjusts its weights and bias
    based on the error between the predicted output and the true output.
    Parameters:
    - neta (float): The learning rate. Default is 0.001.
    - bias (int): The bias term. Default is 0.
    Attributes:
    - bias (int): The bias term.
    - neta (float): The learning rate.
    - errors_ (list): List to store the error values after each update.
    - weights_ (list): List to store the weight values after each update.
    Methods:
    - _initialize_weights(n_features): Initialize the weights randomly.
    - net_input(X): Calculate the net input.
    - activation(X): Apply the activation function.
    - least_mean_squares_error(target, y): Calculate the least mean squares error.
    - learning_rule(X, target): Update the weights and bias based on the learning rule.
    - fit(X, y, epochs): Fit the model to the training data.
    - predict(X): Predict the class labels for the input data.
    - get_errors(): Get the list of error values.
    - get_weights(): Get the list of weight values.
    """
    def __init__(self, neta=0.001, bias=0):
        """
        Initializes an instance of the ADALINE class.

        Parameters:
        - neta (float): The learning rate. Default is 0.001.
        - bias (int): The bias term. Default is 0.
        """
        self.bias = bias
        self.neta = neta
        self.errors_ = [] 
        self.weights_ = [] 

    def _initialize_weights(self, n_features):
        """
        Initializes the weights of the ADALINE model.

        Parameters:
        - n_features (int): The number of input features.

        Returns:
        - None
        """
        self.weights = np.random.rand(n_features)
        self.weights_.append(self.weights.copy())

    def net_input(self, X: np.array) -> float:
        """
        Calculates the net input of the ADALINE model.

        Parameters:
            X (np.array): The input data.

        Returns:
            float: The net input value.
        """
        return np.dot(X, self.weights) + self.bias

    def activation(self, X: np.array) -> float:
        """
        Applies the activation function to the input.

        Parameters:
            X (np.array): The input array.

        Returns:
            float: The output of the activation function.
        """
        return X

    def least_mean_squares_error(self, target: np.array, y: np.array) -> float:
        """
        Calculates the mean squared error between the target and predicted values.

        Parameters:
            target (np.array): The target values.
            y (np.array): The predicted values.

        Returns:
            float: The mean squared error.
        """
        return np.square(target - y).mean()

    def learning_rule(self, X: np.array, target: np.array):
        """
        Updates the weights and bias of the ADALINE model using the delta rule.

        Parameters:
        - X (np.array): The input data.
        - target (np.array): The target values.

        Returns:
        None
        """
        y = self.activation(self.net_input(X))
        error = target - y
        self.weights += self.neta * X * error  # No transpose needed for 1D X
        self.bias += self.neta * error

        # Store error and weights after each update for tracking
        error_value = self.least_mean_squares_error(target, y)
        self.errors_.append(error_value)
        self.weights_.append(self.weights.copy())

    def fit(self, X: np.array, y: int, epochs=100):
        """
        Fits the ADALINE model to the given training data.
        Parameters:
        - X (np.array): The input training data.
        - y (int): The target training data.
        - epochs (int): The number of training epochs (default=100).
        """
        self._initialize_weights(len(X))
        
        for _ in range(epochs):
            self.learning_rule(X, y)

    def predict(self, X: np.array) -> int:
        """
        Predicts the class label for the given input data.

        Parameters:
        X (np.array): The input data to be classified.

        Returns:
        int: The predicted class label (1 or -1).
        """
        net_input = self.net_input(X)
        return np.where(self.activation(net_input) >= 0.0, 1, -1)

    def get_errors(self):
        """
        Returns the list of errors during the training process.

        Returns:
            list: The list of errors.
        """
        return self.errors_

    def get_weights(self):
        """
        Returns the weights of the ADALINE model.

        Returns:
            numpy.ndarray: The weights of the ADALINE model.
        """
        return self.weights_
