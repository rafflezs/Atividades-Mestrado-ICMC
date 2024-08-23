import numpy as np

class InputHandler:
    """
    A class that handles input data.
    Args:
        file_path (str): The path to the input file.
    Attributes:
        _input (numpy.ndarray): The input data loaded from the file.
    Methods:
        flatten_input(): Flattens the input data into a 1-dimensional array.
        reduce_by_sum_input(): Reduces the input data by summing each row.
        get_input(): Returns the input data.
    """

    _input: np.ndarray = None

    def __init__(self, file_path: str) -> None:
        """
        Initializes an instance of the InputHandler class.

        Args:
            file_path (str): The path to the input file.
        """
        self._input = np.loadtxt(file_path, dtype=int)

    def flatten_input(self):
        """
        Flattens the input data into a 1-dimensional array.

        Returns:
            numpy.ndarray: The flattened input data.
        """
        return np.matrix.flatten(self._input)
    
    def reduce_by_sum_input(self):
        """
        Reduces the input data by summing each row.

        Returns:
            list: The reduced input data.
        """
        return [sum(row) for row in self._input]
    
    def get_input(self):
        """
        Returns the input data.

        Returns:
            numpy.ndarray: The input data.
        """
        return self._input