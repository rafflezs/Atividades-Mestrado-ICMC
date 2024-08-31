import numpy as np

class MLP:
    """
    Classe para um Perceptron Multicamadas (MLP).

    Atributos:
    ----------
    W1 : numpy.ndarray
        Pesos da camada de entrada para a camada oculta.
    b1 : numpy.ndarray
        Bias da camada oculta.
    W2 : numpy.ndarray
        Pesos da camada oculta para a camada de saída.
    b2 : numpy.ndarray
        Bias da camada de saída.

    Métodos:
    --------
    __init__(n_inputs, n_hidden, n_output):
        Inicializa os pesos e bias do MLP.
    forward(X):
        Realiza a passagem para frente (forward pass) da rede neural.
    backwardpropagation(y_treino):
        Realiza a retropropagação (backpropagation) para calcular os deltas.
    atualiza_pesos(X, delta_saida, delta_hidden, eta):
        Atualiza os pesos e bias da rede neural.
    sigmoide(x):
        Função de ativação sigmoide.
    treino(X, y, epochs, eta):
        Treina a rede neural por um número especificado de épocas.
    predict(X):
        Faz uma previsão com base na entrada fornecida.
    sigmoide_derivada(x):
        Calcula a derivada da função sigmoide.
    """

    def __init__(self, n_inputs, n_hidden, n_output):
        """
        Inicializa os pesos e bias do MLP.

        Parâmetros:
        -----------
        n_inputs : int
            Número de neurônios na camada de entrada.
        n_hidden : int
            Número de neurônios na camada oculta.
        n_output : int
            Número de neurônios na camada de saída.
        """
        self.W1 = np.random.randn(n_inputs, n_hidden)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output)
        self.b2 = np.zeros((1, n_output))

    def forward(self, X):
        """
        Realiza a passagem para frente (forward pass) da rede neural.

        Parâmetros:
        -----------
        X : numpy.ndarray
            Dados de entrada.

        Retorna:
        --------
        numpy.ndarray
            Saída da rede neural.
        """
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoide(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        output = self.sigmoide(self.z2)
        return output

    def backwardpropagation(self, y_treino):
        """
        Realiza a retropropagação (backpropagation) para calcular os deltas.

        Parâmetros:
        -----------
        y_treino : numpy.ndarray
            Saída prevista pela rede neural.

        Retorna:
        --------
        tuple
            Deltas da camada de saída e da camada oculta.
        """
        custo = self.target - y_treino
        delta_saida = custo * self.sigmoide_derivada(y_treino)
        
        error_hidden_layer = delta_saida.dot(self.W2.T)
        delta_hidden = error_hidden_layer * self.sigmoide_derivada(self.a1)

        return delta_saida, delta_hidden
        
    def atualiza_pesos(self, X, delta_saida, delta_hidden, eta):
        """
        Atualiza os pesos e bias da rede neural.

        Parâmetros:
        -----------
        X : numpy.ndarray
            Dados de entrada.
        delta_saida : numpy.ndarray
            Deltas da camada de saída.
        delta_hidden : numpy.ndarray
            Deltas da camada oculta.
        eta : float
            Taxa de aprendizado.
        """
        self.W2 += self.a1.T.dot(delta_saida) * eta
        self.b2 += np.sum(delta_saida, axis=0, keepdims=True) * eta
        self.W1 += X.T.dot(delta_hidden) * eta
        self.b1 += np.sum(delta_hidden, axis=0, keepdims=True) * eta

    def sigmoide(self, x):
        """
        Função de ativação sigmoide.

        Parâmetros:
        -----------
        x : numpy.ndarray
            Entrada para a função de ativação.

        Retorna:
        --------
        numpy.ndarray
            Saída da função sigmoide.
        """
        return 1 / (1 + np.exp(-x))

    def treino(self, X, y, epochs, eta):
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
        for _ in range(epochs):
            y_treino = self.forward(X)
            delta_saida, delta_hidden = self.backwardpropagation(y_treino)
            self.atualiza_pesos(X=X, eta=eta, delta_saida=delta_saida, delta_hidden=delta_hidden)

    def predict(self, X):
        """
        Faz uma previsão com base na entrada fornecida.

        Parâmetros:
        -----------
        X : numpy.ndarray
            Dados de entrada.

        Retorna:
        --------
        int
            Previsão (1 ou 0).
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.forward(X)

    def sigmoide_derivada(self, x):
        """
        Calcula a derivada da função sigmoide.

        Parâmetros:
        -----------
        x : numpy.ndarray
            Entrada para a função de derivada.

        Retorna:
        --------
        numpy.ndarray
            Derivada da função sigmoide.
        """
        return x * (1 - x)