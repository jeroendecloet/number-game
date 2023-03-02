from abc import ABC, abstractmethod
import numpy as np


class BaseOperation(ABC):

    def __init__(self):
        super().__init__()
        # self.duplicates = True

    def __call__(self, x: np.ndarray, y: np.ndarray, x_str: np.ndarray, y_str: np.ndarray) -> (np.ndarray, np.ndarray):

        # Filter inputs (if any)
        x, y, x_str, y_str = self.filter_inputs(x, y, x_str, y_str)

        # Broadcast x and y
        _x = np.tile(x, (len(y), 1)).flatten()
        _y = np.tile(y.reshape(-1, 1), (1, len(x))).flatten()
        _x_str = np.tile(x_str, (len(y), 1)).flatten()
        _y_str = np.tile(y_str.reshape(-1, 1), (1, len(x))).flatten()

        # Do operations
        results = self.operation(_x, _y)
        results_str = np.array([self.operation_string(a, b) for (a, b) in zip(_x_str, _y_str)])

        # Filter outputs (if any)
        results, results_str = self.filter_outputs(results, results_str)

        # Remove duplicates if activated
        # if self.duplicates:
        #     results, results_str = self.remove_duplicates(results, results_str)

        return results, results_str

    @abstractmethod
    def operation(self, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        pass

    @abstractmethod
    def operation_string(self, x_str: np.ndarray, y_str: np.ndarray) -> (np.ndarray, np.ndarray):
        pass

    def filter_inputs(self, x: np.ndarray, y: np.ndarray, x_str: np.ndarray, y_str: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        return x, y, x_str, y_str

    def filter_outputs(self, results: np.ndarray, results_str: np.ndarray) -> (np.ndarray, np.ndarray):
        return results, results_str

    # def remove_duplicates(self, results: np.ndarray, results_str: np.ndarray) -> (np.ndarray, np.ndarray):
    #     """ When working with duplicate input numbers, there might be duplicates in the calculations. Remove these. """
    #     unique_str, idx = np.unique(results_str, return_index=True)
    #     if len(unique_str) < len(results_str):
    #         return results[idx], unique_str
    #     else:
    #         return results, results_str
    #
    # def set_duplicates(self, duplicates):
    #     self.duplicates = duplicates



class Plus(BaseOperation):

    def __init__(self):
        super().__init__()

    def operation(self, x, y):
        return x + y

    def operation_string(self, x, y):
        return f"({x} + {y})"


class Minus(BaseOperation):
    def __init__(self, invert=False):
        super().__init__()
        self.invert = invert

    def operation(self, x, y):
        if self.invert:
            return y - x
        else:
            return x - y

    def operation_string(self, x, y):
        if self.invert:
            return f"({y} - {x})"
        else:
            return f"({x} - {y})"


class Times(BaseOperation):
    def __init__(self):
        super().__init__()

    def operation(self, x, y):
        return x * y

    def operation_string(self, x, y):
        return f"({x} * {y})"


class Divide(BaseOperation):
    def __init__(self, invert):
        super().__init__()
        self.invert = invert

    def operation(self, x, y):
        if self.invert:
            return y / x
        else:
            return x / y

    def operation_string(self, x, y):
        if self.invert:
            return f"({y} / {x})"
        else:
            return f"({x} / {y})"

    def filter_inputs(self, x: np.ndarray, y: np.ndarray, x_str: np.ndarray, y_str: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """ Filter out division by zero. """
        if self.invert:
            x_str = x_str[x != 0]
            x = x[x != 0]
        else:
            y_str = y_str[y != 0]
            y = y[y != 0]

        return x, y, x_str, y_str
