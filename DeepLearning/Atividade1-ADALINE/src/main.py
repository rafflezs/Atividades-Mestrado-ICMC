from input_handler import InputHandler
from adaline_class import Adaline

if __name__ == "__main__":
    
    clf = Adaline(
        X=InputHandler("data/opposite-of-inverted/5x5-1.txt").flatten_input(),
        bias=0,
        fill_method='random'
        )
    
    