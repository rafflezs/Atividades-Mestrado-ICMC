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
    plt.savefig(f"../figures/{file_name}") 
    plt.close()

def debug_network(X, y, error, weights):
    print(f"X:\t\t{X}")
    print(f"y:\t\t{y}")
    print(f"Error:\t\t{error[-1]}")
    print(f"Weights:\t{weights[-1]}")

def test_network(label, file_name, clf):
    X_test = InputHandler(file_path=f"../data/label{label}/5x5-{file_name}.txt").flatten_input()
    prediction = clf.predict(X=X_test)
    print(f"5x5-{file_name} prediction:\t{prediction}")

if __name__ == "__main__":

    clf = Adaline(neta=0.001, bias=0)
    
    label_train = "+1"
    X_train = InputHandler(file_path=f"../data/label{label_train}/5x5-train.txt").flatten_input()
    y_train = int(label_train)
    
    clf.fit(X=X_train, y=y_train, epochs=1000)

    errors = clf.get_errors()
    weights = clf.get_weights()
    weights_array = np.array(weights)

    plot_figure(errors, 'Mean Squared Error per Epoch', 'Epoch', 'Mean Squared Error', 'error_plot.png')
    plot_figure(weights, 'Weights Evolution', 'Epoch', 'Weights Value', 'weights_plot.png')

    debug_network(X_train, y_train, errors, weights)

    for label_test in ["+1", "-1"]:
        print(f"\nExpected:\t\t{int(label_test)}")
        test_network(label_test, "train", clf)
        for file_num in range(1, 7):
            test_network(label_test, f"test{file_num}", clf)
