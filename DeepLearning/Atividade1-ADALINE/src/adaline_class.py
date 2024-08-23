import numpy as np

np.random.seed(13)

class Adaline:

    X: np.array = None
    weights: np.array = None
    bias: int = None

    def __init__(self, X: np.array, target, neta = 0.001, fill_method: str = 'random', bias: int = 0):
        
        self.X = X
        self.target = target
        self.bias = bias

        if fill_method == 'random':
            self.weights = np.random.uniform(0, 1, len(X))
        elif fill_method == 'zero':
            self.weights = np.zeros(len(X))
        elif fill_method == 'one':
            self.weights = np.ones(len(X))

    def net_input(self, x: np.array):
        self.y = np.dot(x, self.weights) + self.bias
    
    def least_mean_squares_error(self):

        # in case of smaller ints, use x*x instead of x**2
        # though np.square is faster than both
        self.E = np.square(self.target - self.y)

    def learning_rule(self):
        self.weights += self.neta * self.X * (self.target - self.y)