import os
import sys
import unittest
import numpy as np

sys.path.append("src")
from input_handler import InputHandler

class TestInputHandler(unittest.TestCase):

    def setUp(self):
        self.file_path = "tests/dump/inputhandlertest.txt"
        self.input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        np.savetxt(self.file_path, self.input_data, fmt="%d")

    def tearDown(self):
        os.remove(self.file_path)

    def test_init(self):
        input_handler = InputHandler(self.file_path)
        np.testing.assert_array_equal(input_handler.get_input(), self.input_data)

    def test_flatten_input(self):
        input_handler = InputHandler(self.file_path)
        flattened_input = input_handler.flatten_input()
        expected_output = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        np.testing.assert_array_equal(flattened_input, expected_output)

    def test_reduce_by_sum_input(self):
        input_handler = InputHandler(self.file_path)
        reduced_input = input_handler.reduce_by_sum_input()
        expected_output = [6, 15, 24]
        self.assertEqual(reduced_input, expected_output)

if __name__ == "__main__":
    unittest.main()