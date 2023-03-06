from abc import ABC, abstractmethod
import numpy as np


class BaseOperation(ABC):
    """ Base operation """
    names = list

    def __init__(self):
        super().__init__()

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


class Addition(BaseOperation):
    """ Addition operator """
    names = ['addition', 'add', 'plus']

    def __init__(self):
        super().__init__()

    def operation(self, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        return x + y

    def operation_string(self, x_str: np.ndarray, y_str: np.ndarray) -> (np.ndarray, np.ndarray):
        return f"({x_str} + {y_str})"


class Subtraction(BaseOperation):
    """ Subtraction operator """
    names = ['subtraction', 'minus', 'min']

    def __init__(self, invert=False):
        super().__init__()
        self.invert = invert

    def operation(self, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        if self.invert:
            return y - x
        else:
            return x - y

    def operation_string(self, x_str: np.ndarray, y_str: np.ndarray) -> (np.ndarray, np.ndarray):
        if self.invert:
            return f"({y_str} - {x_str})"
        else:
            return f"({x_str} - {y_str})"


class Multiplication(BaseOperation):
    """ Multiplication operator """
    names = ['multiplication', 'times']

    def __init__(self):
        super().__init__()

    def operation(self, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        return x * y

    def operation_string(self, x_str: np.ndarray, y_str: np.ndarray) -> (np.ndarray, np.ndarray):
        return f"({x_str} * {y_str})"


class Division(BaseOperation):
    """ Division operator """
    names = ['division', 'divide']

    def __init__(self, invert=False):
        super().__init__()
        self.invert = invert

    def operation(self, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        if self.invert:
            return y / x
        else:
            return x / y

    def operation_string(self, x_str: np.ndarray, y_str: np.ndarray) -> (np.ndarray, np.ndarray):
        if self.invert:
            return f"({y_str} / {x_str})"
        else:
            return f"({x_str} / {y_str})"

    def filter_inputs(self, x: np.ndarray, y: np.ndarray, x_str: np.ndarray, y_str: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """ Filter out division by zero. """
        if self.invert:
            x_str = x_str[x != 0]
            x = x[x != 0]
        else:
            y_str = y_str[y != 0]
            y = y[y != 0]

        return x, y, x_str, y_str


class Power(BaseOperation):
    """ Power operation """
    names = ['power']

    def __init__(self, invert=False):
        super().__init__()
        self.invert = invert

    def operation(self, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        try:
            if self.invert:
                return y ** x
            else:
                return x ** y
        except OverflowError:
            print(x, y, self.invert)
        except RuntimeWarning:
            print(x, y, self.invert)

    def operation_string(self, x_str: np.ndarray, y_str: np.ndarray) -> (np.ndarray, np.ndarray):
        if self.invert:
            return f"({y_str} ^ {x_str})"
        else:
            return f"({x_str} ^ {y_str})"

    def filter_inputs(self, x: np.ndarray, y: np.ndarray, x_str: np.ndarray, y_str: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """ Filter out bases and powers smaller than 0.01 or larger than 100. """
        _min = 1e-2
        _max = 1e2
        x_str = x_str[(_min < x) & (x < _max)]
        x = x[(_min < x) & (x < _max)]
        y_str = y_str[(_min < y) & (y < _max)]
        y = y[(_min < y) & (y < _max)]

        return x, y, x_str, y_str

    def filter_outputs(self, results: np.ndarray, results_str: np.ndarray) -> (np.ndarray, np.ndarray):
        """ Filter out results smaller than 1e-10 and larger than 1e10. """
        _min = 1e-10
        _max = 1e10
        results_str = results_str[(_min < results) & (results < _max)]
        results = results[(_min < results) & (results < _max)]
        return results, results_str
