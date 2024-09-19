import numpy as np

class MLP:
    """
    Classe para um Perceptron Multicamadas (MLP).

    Atributos:
    ----------
    weights : numpy.ndarray
        Pesos entre cada camada (entrada, ocultas e saída).
    biases : numpy.ndarray
        Vieses das camadas ocultas e de saída.
    target : numpy.ndarray
        Dados de saída esperados.
    deltas : numpy.ndarray
        Deltas das camadas ocultas e de saída.
    eta : float
        Taxa de aprendizado.

    Métodos:
    --------
    __init__(n_input, n_hidden, n_output):
        Inicializa os pesos e vieses do MLP.
    function_hidden(x):
        Função de ativação das camadas ocultas e sua respectiva derivada.
    function_output(x):
        Função de ativação da camada de saída e sua respectiva derivada.
    forward_pass(X):
        Realiza a passagem para frente (forward pass) da rede neural.
    backpropagation(outputs):
        Realiza a retropropagação (backpropagation) para calcular os deltas.
    update_weights(X, outputs):
        Atualiza os pesos e vieses da rede neural.
    train(X, y, epochs, eta):
        Treina a rede neural por um número especificado de épocas.
    predict(X):
        Faz uma previsão com base na entrada fornecida.
    """
    
    def __init__(self, n_input, n_hidden, n_output, momentum=0.1 ,function_hidden="relu", function_output="sigmoid"):
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
            "sigmoid": (
                lambda x: 1 / (1 + np.exp(-x)),
                lambda x: x * (1 - x)
            ),
            "relu": (
                lambda x: np.maximum(0, x),
                lambda x: np.where(x > 0, 1, 0)
            )
        }

        # print("\nArgument check")
        # print(f"N input: {n_input} | N hidden: {n_hidden} | N output: {n_output} | Momentum: {momentum}")


        # print("\nWeights check")
        self.weights = [np.random.randn(n_input, n_hidden[0])]
        self.biases = [np.zeros((1, n_hidden[0]))]
        
        # print(f"Weight shape: {self.weights[0].shape} | Bias shape: {self.biases[0].shape}")
        # print(f"Weight: {self.weights[0]} | Bias: {self.biases[0]}")

        # print("\nMomentun check")
        self.momentum = momentum
        self.velocity_weights = [np.zeros_like(w) for w in self.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.biases]
        # print(f"Momentum: {self.momentum} | Velocity Weights Shape: {self.velocity_weights[0].shape} | Velocity Biases Shape: {self.velocity_biases[0].shape}")
        # print(f"Velocity Weights: {self.velocity_weights[0]} | Velocity Biases: {self.velocity_biases[0]}")


        # print("Hidden layer check (Bias and Weights)")
        for layer in range(1, len(n_hidden)):
            self.weights.append(np.random.randn(n_hidden[layer - 1], n_hidden[layer]))
            self.biases.append(np.zeros((1, n_hidden[layer])))
            # print(f"Layer: {layer} | Weight shape: {self.weights[layer].shape} | Bias shape: {self.biases[layer].shape}")
            # print(f"Layer: {layer} | Weight: {self.weights[layer]} | Bias: {self.biases[layer]}")

        # print("\nOutput layer check (Bias and Weights)")
        self.weights.append(np.random.randn(n_hidden[-1], n_output))
        self.biases.append(np.zeros((1, n_output)))
        # print(f"Output layer | Weight shape: {self.weights[-1].shape} | Bias shape: {self.biases[-1].shape}")
        # print(f"Output layer | Weight: {self.weights[-1]} | Bias: {self.biases[-1]}")

        # print("\nWeights and Biases check")
        self.weights = np.array(self.weights, dtype=object)
        # self.biases = np.array(self.biases, dtype=object)
        # print(f"Weights: {self.weights} | Biases: {self.biases}")

        # print("\nVelocities check")
        self.velocity_weights = [np.zeros_like(w) for w in self.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.biases]
        # print(f"Velocity Weights Shape: {self.velocity_weights[0].shape} | Velocity Biases Shape: {self.velocity_biases[0].shape}")
        
        self.function_hidden = activation_functions[function_hidden]
        self.function_output = activation_functions[function_output]


    def forward_pass(self, X):
        """
        Realiza a passagem para frente (forward pass) da rede neural.

        Parâmetros:
        -----------
        X : numpy.ndarray
            Dados de entrada.

        Retorna:
        --------
        numpy.ndarray
            Saídas das camadas ocultas e de saída.
        """
        outputs = [self.function_hidden[0](np.dot(X, self.weights[0]) + self.biases[0])]

        for layer in range(1, len(self.weights) - 1):
            weighted_sum = np.dot(outputs[-1], self.weights[layer]) + self.biases[layer]
            outputs.append(self.function_hidden[0](weighted_sum))

        final_output = np.dot(outputs[-1], self.weights[-1]) + self.biases[-1]
        outputs.append(self.function_output[0](final_output))

        return outputs

    def backpropagation(self, outputs):
        """
        Realiza a retropropagação (backpropagation) para calcular os deltas.

        Parâmetros:
        -----------
        outputs : numpy.ndarray
            Saídas das camadas ocultas e de saída.
        """
        loss = self.target - outputs[-1]
        deltas = [loss * self.function_output[1](outputs[-1])]

        for layer in range(len(self.weights) - 2, -1, -1):
            error = np.dot(deltas[-1], self.weights[layer + 1].T)
            deltas.append(error * self.function_hidden[1](outputs[layer]))

        self.deltas = list(reversed(deltas))

    def update_weights(self, X, outputs):
        """
        Atualiza os pesos e vieses da rede neural com momentum.

        Parâmetros:
        -----------
        X : numpy.ndarray
            Dados de entrada.
        outputs : numpy.ndarray
            Saídas das camadas ocultas e de saída.
        """
        # Atualização para a última camada
        delta_w = np.dot(outputs[-2].T, self.deltas[-1]) * self.eta
        delta_b = np.sum(self.deltas[-1], axis=0, keepdims=True) * self.eta

        # Atualiza as velocidades (momentum)
        self.velocity_weights[-1] = self.momentum * self.velocity_weights[-1] + delta_w
        self.velocity_biases[-1] = self.momentum * self.velocity_biases[-1] + delta_b

        # Aplica a atualização aos pesos e vieses
        self.weights[-1] += self.velocity_weights[-1]
        self.biases[-1] += self.velocity_biases[-1]

        # Atualização para as camadas ocultas
        for layer in range(len(self.weights) - 2, 0, -1):
            delta_w = np.dot(outputs[layer - 1].T, self.deltas[layer]) * self.eta
            delta_b = np.sum(self.deltas[layer], axis=0, keepdims=True) * self.eta

            self.velocity_weights[layer] = self.momentum * self.velocity_weights[layer] + delta_w
            self.velocity_biases[layer] = self.momentum * self.velocity_biases[layer] + delta_b

            self.weights[layer] += self.velocity_weights[layer]
            self.biases[layer] += self.velocity_biases[layer]

        # Atualização para a camada de entrada
        delta_w = np.dot(X.T, self.deltas[0]) * self.eta
        delta_b = np.sum(self.deltas[0], axis=0, keepdims=True) * self.eta

        self.velocity_weights[0] = self.momentum * self.velocity_weights[0] + delta_w
        self.velocity_biases[0] = self.momentum * self.velocity_biases[0] + delta_b

        self.weights[0] += self.velocity_weights[0]
        self.biases[0] += self.velocity_biases[0]

    def train(self, X, y, epochs, eta):
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
        """
        self.target = y
        self.eta = eta
        for _ in range(epochs):
            outputs = self.forward_pass(X)
            self.backpropagation(outputs)
            self.update_weights(X, outputs)

    def predict(self, X):
        """
        Faz uma previsão com base na entrada fornecida.

        Parâmetros:
        -----------
        X : numpy.ndarray
            Dados de entrada.

        Retorna:
        --------
        numpy.ndarray
            Previsão da rede.
        """
        return self.forward_pass(X if X.ndim > 1 else X.reshape(1, -1))[-1]