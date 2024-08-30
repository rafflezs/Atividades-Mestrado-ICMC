import numpy as np

class MultiLayerPerceptron:
    
    def __init__(self, input_X: np.array, output_Y: np.array, hidden_layers: np.array = [16,16], learning_rate: float = 0.1, epochs: int = 1000):
        self.input_X = input_X
        self.output_Y = output_Y
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self.layers = []
        self.errors = []
        self.outputs = []
        self.deltas = []
        self.initialize_weights()

    def initialize_weights(self):
        self.weights = []
        self.biases = []
        self.layers = []
        self.errors = []
        self.outputs = []
        self.deltas = []
        self.layers.append(self.input_X.shape[1])
        self.layers.extend(self.hidden_layers)
        self.layers.append(self.output_Y.shape[1])
        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.rand(self.layers[i], self.layers[i+1]))
            self.biases.append(np.random.rand(self.layers[i+1]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def feed_forward(self, x):
        self.outputs = []
        self.outputs.append(x)
        for i in range(len(self.weights)):
            self.outputs.append(self.sigmoid(np.dot(self.outputs[i], self.weights[i]) + self.biases[i]))
        return self.outputs[-1]
    
    def back_propagation(self, x, y):
        self.feed_forward(x)
        self.errors = []
        self.deltas = []
        self.errors.append(y - self.outputs[-1])
        self.deltas.append(self.errors[-1] * self.outputs[-1] * (1 - self.outputs[-1]))
        for i in range(len(self.weights) - 1, 0, -1):
            self.errors.append(np.dot(self.deltas[0], self.weights[i].T))
            self.deltas.insert(0, self.errors[0] * self.outputs[i] * (1 - self.outputs[i]))
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * np.dot(self.outputs[i].T, self.deltas[i])
            self.biases[i] += self.learning_rate * np.sum(self.deltas[i])

    def train(self):
        for i in range(self.epochs):
            for j in range(self.input_X.shape[0]):
                self.back_propagation(self.input_X[j], self.output_Y[j])

    def predict(self, x):
        return self.feed_forward(x)
    
    def evaluate(self, x, y):
        return np.mean(np.round(self.predict(x)) == y)
    
    def evaluate_all(self, x, y):
        return np.mean([self.evaluate(x[i], y[i]) for i in range(x.shape[0])])
    
    def get_weights(self):
        return self.weights
    
    def get_biases(self):
        return self.biases
    
    def get_layers(self):
        return self.layers
    
    def get_errors(self):
        return self.errors
    
    def get_outputs(self):
        return self.outputs
    
    def get_deltas(self):
        return self.deltas
