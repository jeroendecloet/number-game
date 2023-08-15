from abc import ABC, abstractmethod

import numpy as np


class BaseOperationOne(ABC):
    """ Base operation for one number """
    names = list
    str_format: str

    def __init__(self):
        super().__init__()

    def __call__(self, x: np.ndarray, x_str: np.ndarray) -> (np.ndarray, np.ndarray):

        # Filter inputs (if required)
        x, x_str = self.filter_inputs(x, x_str)

        # Do operation
        results = self.operation(x)
        results_str = np.array([self.operation_string(_x_str) for _x_str in x_str])

        # Filter outputs (if required)
        results, results_str = self.filter_outputs(results, results_str)

        return results, results_str

    @abstractmethod
    def operation(self, x: np.ndarray) -> np.ndarray:
        pass

    def operation_string(self, x_str: np.ndarray) -> str:
        return self.str_format.format(x_str=x_str)

    def filter_inputs(self, x: np.ndarray, x_str: np.ndarray) -> (np.ndarray, np.ndarray):
        return x, x_str

    def filter_outputs(self, results: np.ndarray, results_str: np.ndarray) -> (np.ndarray, np.ndarray):
        return results, results_str


class BaseOperationTwo(ABC):
    """ Base operation for two numbers """
    names = list
    str_format: str

    def __init__(self):
        super().__init__()

    def __call__(self, x: np.ndarray, y: np.ndarray, x_str: np.ndarray, y_str: np.ndarray) -> (np.ndarray, np.ndarray):

        # Filter inputs (if required)
        x, y, x_str, y_str = self.filter_inputs(x, y, x_str, y_str)

        # Broadcast x and y
        _x = np.tile(x, (len(y), 1)).flatten()
        _y = np.tile(y.reshape(-1, 1), (1, len(x))).flatten()
        _x_str = np.tile(x_str, (len(y), 1)).flatten()
        _y_str = np.tile(y_str.reshape(-1, 1), (1, len(x))).flatten()

        # Do operations
        results = self.operation(_x, _y)
        results_str = np.array([self.operation_string(a, b) for (a, b) in zip(_x_str, _y_str)])

        # Filter outputs (if required)
        results, results_str = self.filter_outputs(results, results_str)

        return results, results_str

    @abstractmethod
    def operation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    def operation_string(self, x_str: np.ndarray, y_str: np.ndarray) -> str:
        return self.str_format.format(x_str=x_str, y_str=y_str)

    def filter_inputs(self, x: np.ndarray, y: np.ndarray, x_str: np.ndarray, y_str: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        return x, y, x_str, y_str

    def filter_outputs(self, results: np.ndarray, results_str: np.ndarray) -> (np.ndarray, np.ndarray):
        return results, results_str


class Addition(BaseOperationTwo):
    """ Addition operator """
    names = ['addition', 'add', 'plus']
    str_format = "({x_str} + {y_str})"

    def operation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x + y


class Subtraction(BaseOperationTwo):
    """ Subtraction operator """
    names = ['subtraction', 'minus', 'min']
    str_format = "({x_str} - {y_str})"

    def __init__(self, invert=False):
        super().__init__()
        self.invert = invert

    def operation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.invert:
            return y - x
        else:
            return x - y

    def operation_string(self, x_str: np.ndarray, y_str: np.ndarray) -> str:
        if self.invert:
            return self.str_format.format(x_str=y_str, y_str=x_str)
        else:
            return self.str_format.format(x_str=x_str, y_str=y_str)


class Multiplication(BaseOperationTwo):
    """ Multiplication operator """
    names = ['multiplication', 'times']
    str_format = "({x_str} * {y_str})"

    def __init__(self):
        super().__init__()

    def operation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x * y


class Division(BaseOperationTwo):
    """ Division operator """
    names = ['division', 'divide']
    str_format = "({x_str} / {y_str})"

    def __init__(self, invert=False):
        super().__init__()
        self.invert = invert

    def operation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.invert:
            return y / x
        else:
            return x / y

    def operation_string(self, x_str: np.ndarray, y_str: np.ndarray) -> str:
        if self.invert:
            return self.str_format.format(x_str=y_str, y_str=x_str)
        else:
            return self.str_format.format(x_str=x_str, y_str=y_str)

    def filter_inputs(self, x: np.ndarray, y: np.ndarray, x_str: np.ndarray, y_str: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """ Filter out division by zero. """
        if self.invert:
            x_str = x_str[x != 0]
            x = x[x != 0]
        else:
            y_str = y_str[y != 0]
            y = y[y != 0]

        return x, y, x_str, y_str


class Power(BaseOperationTwo):
    """ Power operation """
    names = ['power']
    str_format = "({x_str} ^ {y_str})"

    def __init__(self, invert=False):
        super().__init__()
        self.invert = invert

    def operation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        try:
            if self.invert:
                return y ** x
            else:
                return x ** y
        except OverflowError:
            print(x, y, self.invert)
        except RuntimeWarning:
            print(x, y, self.invert)

    def operation_string(self, x_str: np.ndarray, y_str: np.ndarray) -> str:
        if self.invert:
            return self.str_format.format(x_str=y_str, y_str=x_str)
        else:
            return self.str_format.format(x_str=x_str, y_str=y_str)

    def filter_inputs(self, x: np.ndarray, y: np.ndarray, x_str: np.ndarray, y_str: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """ Filter out bases and powers smaller than 0.01 or larger than 100. """
        _min = 1e-2
        _max = 1e2
        mask_x = (_min < x) & (x < _max)
        mask_y = (_min < y) & (y < _max)

        x_str = x_str[mask_x]
        x = x[mask_x]
        y_str = y_str[mask_y]
        y = y[mask_y]
        return x, y, x_str, y_str

    def filter_outputs(self, results: np.ndarray, results_str: np.ndarray) -> (np.ndarray, np.ndarray):
        """ Filter out results smaller than 1e-10 and larger than 1e10. """
        _min = 1e-10
        _max = 1e10
        mask = (_min < results) & (results < _max)

        results_str = results_str[mask]
        results = results[mask]
        return results, results_str


class SquareRoot(BaseOperationOne):
    """ Square root operation """
    names = ['sqrt']
    str_format = "sqrt({x_str})"

    def operation(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(x)

    def filter_inputs(self, x: np.ndarray, x_str: np.ndarray) -> (np.ndarray, np.ndarray):
        positive = x > 0
        _is_not_one = x != 1
        mask = positive & _is_not_one

        x_str = x_str[mask]
        x = x[mask]
        return x, x_str


class Factorial(BaseOperationOne):
    """ Factorial operation """
    names = ['factorial', '!']
    str_format = "{x_str}!"

    def operation(self, x: np.ndarray) -> np.ndarray:
        return np.array([np.prod(np.arange(1, _x + 1)) for _x in x])

    def filter_inputs(self, x: np.ndarray, x_str: np.ndarray) -> (np.ndarray, np.ndarray):
        positive = x > 0
        _is_int = (x == np.rint(x))
        _is_not_one = x != 1
        _max = 15
        _mask = positive & _is_int & _is_not_one & (x < _max)

        x_str = x_str[_mask]
        x = x[_mask]
        return x, x_str

    def filter_outputs(self, results: np.ndarray, results_str: np.ndarray) -> (np.ndarray, np.ndarray):
        _min = 1e-10
        _max = 1e10
        results_str = results_str[(_min < results) & (results < _max)]
        results = results[(_min < results) & (results < _max)]
        return results, results_str
