import numpy as np

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',',dtype=int)
    X = np.array(data[:, :-1])  # Todas as colunas exceto a última
    y = np.array(data[:, -1]).reshape(-1, 1)   # Última coluna
    return X, y