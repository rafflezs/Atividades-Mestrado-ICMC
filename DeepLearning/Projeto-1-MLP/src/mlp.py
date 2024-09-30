import numpy as np

class MLP:
    """
    Classe para um Perceptron Multicamadas (MLP).

    Atributos:
    ----------
    weights : list
        Pesos entre cada camada (entrada, ocultas e saída).
    biases : list
        Vieses das camadas ocultas e de saída.
    velocity_weights : list
        Velocidade de atualização dos pesos.
    velocity_biases : list
        Velocidade de atualização dos vieses.
    target : numpy.ndarray
        Dados de saída esperados.
    deltas : numpy.ndarray
        Deltas das camadas ocultas e de saída.
    eta : float
        Taxa de aprendizado.
    momentum : float
        Velocidade de aprendizado.

    Métodos:
    --------
    __init__(n_input, n_hidden, n_output, function_hidden="relu", function_output="sigmoid"):
        Inicializa os pesos e vieses do MLP.
    function_hidden(x):
        Função de ativação das camadas ocultas e sua respectiva derivada.
    function_output(x):
        Função de ativação da camada de saída e sua respectiva derivada.
    forward_pass(X):
        Realiza a passagem para frente (forward pass) da rede neural.
    backpropagation(index, outputs):
        Realiza a retropropagação (backpropagation) para calcular os deltas.
    update_weights(X, outputs):
        Atualiza os pesos e vieses da rede neural com momentum.
    train(X, y, epochs, eta, momentum):
        Treina a rede neural por um número especificado de épocas.
    predict(X):
        Faz previsões com base na entrada fornecida.
    """

    def __init__(self, n_input, n_hidden, n_output, function_hidden="relu", function_output="sigmoid"):
        """
        Inicializa os pesos e vieses do MLP.

        Parâmetros:
        -----------
        n_input : int
            Número de neurônios na camada de entrada.
        n_hidden : list ou numpy.ndarray
            Número de neurônios em cada camada oculta.
        n_output : int
            Número de neurônios na camada de saída.
        function_hidden : str
            Função de ativação das camadas ocultas
        function_output: str
            Função de ativação da camada de saída
        """
        activation_functions = {
            "relu": (
                lambda x: np.maximum(0, x),
                lambda x: np.where(x > 0, 1, 0)
            ),
            "sigmoid": (
                lambda x: 1 / (1 + np.exp(-x)),
                lambda x: x * (1 - x)
            ),
            "softmax": (
                lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True),
                lambda x: np.diagflat(x) - np.outer(x, x)
            )
        }

        if len(n_hidden) >= 1:
            # Inicializa a primeira camada oculta
            self.weights = [np.random.randn(n_input, n_hidden[0])]
            self.biases = [np.zeros((1, n_hidden[0]))]

            # Inicializa as demais camadas ocultas
            for layer in range(1, len(n_hidden)):
                self.weights.append(np.random.randn(n_hidden[layer - 1], n_hidden[layer]))
                self.biases.append(np.zeros((1, n_hidden[layer])))

            # Inicializa a camada de saída
            self.weights.append(np.random.randn(n_hidden[-1], n_output))
            self.biases.append(np.zeros((1, n_output)))

        else:
            # Inicializa a camada de saída
            self.weights = [np.random.randn(n_input, n_output)]
            self.biases = [np.zeros((1, n_output))]

        self.function_hidden = activation_functions[function_hidden]
        self.function_output = activation_functions[function_output]
        self.velocity_weights = [np.zeros_like(w) for w in self.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.biases]

    def forward_pass(self, X):
        """
        Realiza a passagem para frente (forward pass) da rede neural.

        Parâmetros:
        -----------
        X : numpy.ndarray
            Dados de entrada.

        Retorna:
        --------
        list
            Saídas das camadas ocultas e de saída.
        """
        # Saída da primeira camamada oculta
        outputs = [self.function_hidden[0](np.dot(X, self.weights[0]) + self.biases[0])]

        # Saída das demais camamadas ocultas
        for layer in range(1, len(self.weights) - 1):
            weighted_sum = np.dot(outputs[-1], self.weights[layer]) + self.biases[layer]
            outputs.append(self.function_hidden[0](weighted_sum))

        # Saída da camada de saída 
        y = np.dot(outputs[-1], self.weights[-1]) + self.biases[-1]
        outputs.append(self.function_output[0](y))

        return outputs

    def backpropagation(self, index, outputs):
        """
        Realiza a retropropagação (backpropagation) para calcular os deltas.

        Parâmetros:
        -----------
        index : int
            Índice da linha da saída esperada
        outputs : list
            Saídas das camadas ocultas e de saída.
        """
        # Calcula o delta da camada de saída
        loss = self.target[index] - outputs[-1]
        deltas = [loss * self.function_output[1](outputs[-1])]

        # Calcula o delta das camadas ocultas
        for layer in range(len(self.weights) - 1, 0, -1):
            error = np.dot(deltas[-1], self.weights[layer].T)
            deltas.append(error * self.function_hidden[1](outputs[layer - 1]))

        self.deltas = list(reversed(deltas))

    def update_weights(self, X, outputs):
        """
        Atualiza os pesos e vieses da rede neural com momentum.

        Parâmetros:
        -----------
        X : numpy.ndarray
            Dados de entrada.
        outputs : list
            Saídas das camadas ocultas e de saída.
        """
        # Atualiza as camadas de saída e ocultas
        for layer in range(len(self.weights) - 1, 0, -1):
            delta_w = np.dot(outputs[layer - 1].T, self.deltas[layer]) * self.eta
            delta_b = np.sum(self.deltas[layer], axis=0, keepdims=True) * self.eta

            # Atualiza as velocidades (momentum)
            self.velocity_weights[layer] = self.momentum * self.velocity_weights[layer] + delta_w
            self.velocity_biases[layer] = self.momentum * self.velocity_biases[layer] + delta_b

            # Aplica a atualização aos pesos e vieses
            self.weights[layer] += self.velocity_weights[layer]
            self.biases[layer] += self.velocity_biases[layer]

        # Atualiza a camada de entrada
        delta_w = np.dot(X.T, self.deltas[0]) * self.eta
        delta_b = np.sum(self.deltas[0], axis=0, keepdims=True) * self.eta

        # Atualiza as velocidades (momentum)
        self.velocity_weights[0] = self.momentum * self.velocity_weights[0] + delta_w
        self.velocity_biases[0] = self.momentum * self.velocity_biases[0] + delta_b

        # Aplica a atualização aos pesos e vieses
        self.weights[0] += self.velocity_weights[0]
        self.biases[0] += self.velocity_biases[0]

    def train(self, X, y, epochs, eta, momentum):
        """
        Treina a rede neural por um número especificado de épocas.

        Parâmetros:
        -----------
        X : numpy.ndarray
            Dados de entrada.
        y : numpy.ndarray
            Dados de saída esperados.
        epochs : int
            Número de épocas de treinamento.
        eta : float
            Taxa de aprendizado.
        momentum : float
            Velocidade de aprendizado.
        """
        self.target = y
        self.eta = eta
        self.momentum = momentum

        for _ in range(epochs):
            for index, row in X.iterrows():
                row = row.to_frame().T
                outputs = self.forward_pass(row)
                self.backpropagation(index, outputs)
                self.update_weights(row, outputs)

    def predict(self, X):
        """
        Faz previsões com base na entrada fornecida.

        Parâmetros:
        -----------
        X : numpy.ndarray
            Dados de entrada.

        Retorna:
        --------
        numpy.ndarray
            Previsões da rede.
        """
        outputs = []
        for _, row in X.iterrows():
            row = row.to_frame().T
            y = self.forward_pass(row)[-1][0]
            outputs.append(y)

        return np.array(outputs)
