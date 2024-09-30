import numpy as np
import tensorflow as tf
from tensorflow import keras

class MLP:
    def __init__(self, X_dim, hidden_layers=[{'units': 10, 'activation': 'relu'}], y_dim=1, **kwargs):
        self.X_dim = X_dim
        self.hidden_layers = hidden_layers
        self.y_dim = y_dim

        print(hidden_layers)

        if 'momentum' in kwargs:
            self.momentum = kwargs.get('momentum', 0.9)
        if 'lr' in kwargs:
            self.lr = kwargs.get('lr', 0.01)

        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(self.X_dim,)))
        for layer in self.hidden_layers:
            model.add(keras.layers.Dense(units=layer['units'], activation=layer['activation']))
        model.add(keras.layers.Dense(units=self.y_dim, activation='linear'))
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=self.lr, momentum=self.momentum), loss='mean_squared_error')
        return model
    
    def fit(self, X, y, **kwargs):
        self.history = None
        if 'epochs' in kwargs:
            self.epochs = kwargs.get('epochs', 100)
        self.history = (self.model.fit(X, y, epochs=self.epochs, batch_size=kwargs.get('batch_size', 32), verbose=kwargs.get('verbose', 1)))

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        return self.model.evaluate(X, y)
    
    def get_summary(self):
        return self.model.summary()
    
    def get_history(self):
        return self.history