import numpy as np

class InputHandler:

    def __init__(self, file_path: str) -> None:
        self._input = np.loadtxt(file_path, dtype=int)

    def flatten_input(self):
        return np.matrix.flatten(self._input)
    
    def reduce_by_sum_input(self):
        return [sum(row) for row in self._input]
    
    def get_input(self):
        return self._input