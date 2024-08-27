import numpy as np
from input_handler import InputHandler
from adaline_class import Adaline
import matplotlib.pyplot as plt

def plot_figure(values, title, xlabel, ylabel, file_name):
    plt.figure()
    plt.plot(values)
    plt.title(title)
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.savefig(f"figures/{file_name}") 
    plt.close()

def debug_network(X, y, error, weights):
    print(X)
    print(y)
    print(error[-1])
    print(weights[-1])

if __name__ == "__main__":

    clf = Adaline(neta=0.001, bias=0)
    
    X_train = InputHandler(file_path="data/label-1-positive/5x5-1.txt").flatten_input()
    y_train = -1
    
    clf.fit(X=X_train, y=y_train, epochs=1000)

    errors = clf.get_errors()
    weights = clf.get_weights()
    weights_array = np.array(weights)

    plot_figure(errors, 'Mean Squared Error per Epoch', 'Epoch', 'Mean Squared Error', 'error_plot.png')
    plot_figure(weights, 'Weights Evolution', 'Epoch', 'Weights Value', 'weights_plot.png')

    debug_network(X_train, y_train, errors, weights)

    X_test = InputHandler(file_path="data/label-1-positive/5x5-7.txt").flatten_input()
    prediction = clf.predict(X=X_test)
    print("Prediction:", prediction)
