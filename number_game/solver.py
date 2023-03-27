from inspect import isclass
from typing import Union
from itertools import combinations

import numpy as np

import operations as ops


def get_operations(operation_names: list[str]) -> (list[callable], list[callable]):
    """
    Translates a list of operation names to actual instanced operations.

    Inputs
    ------
    operation_names: list
        List of names of operations to use, e.g. `plus` or `multiplication`

    Outputs
    -------
    operations_one: list
        List with instanced/callable operations with one parameter
    operations_two: list
        List with instanced/callable operations with two parameters
    """
    all_operation_one_names = [_class for _class in dir(ops) if (isclass(getattr(ops, _class))) and (_class != 'BaseOperationOne') and issubclass(getattr(ops, _class), ops.BaseOperationOne)]
    all_operation_two_names = [_class for _class in dir(ops) if (isclass(getattr(ops, _class))) and (_class != 'BaseOperationTwo') and issubclass(getattr(ops, _class), ops.BaseOperationTwo)]

    # Make a list of all aliases of the operations
    aliases_one = dict()
    for name in all_operation_one_names:
        op = getattr(ops, name)
        aliases_one = {**aliases_one, **dict(zip(op.names, [name] * len(op.names)))}
    aliases_two = dict()
    for name in all_operation_two_names:
        op = getattr(ops, name)
        aliases_two = {**aliases_two, **dict(zip(op.names, [name] * len(op.names)))}

    # Look through the operations to
    operations_one = list()
    operations_two = list()
    active_operations = list()
    for name in operation_names:
        # I
        if name in aliases_one:
            if aliases_one[name] in active_operations:
                print(f"Operation {name} already found! Skipping...")
                continue
            op = getattr(ops, aliases_one[name])

            operations_one.append(op())

            active_operations.append(aliases_one[name])

        elif name in aliases_two:
            # Check if the operation has not been added already
            if aliases_two[name] in active_operations:
                print(f"Operation {name} already found! Skipping...")
                continue

            op = getattr(ops, aliases_two[name])
            # Initialize operations
            if 'invert' in op.__init__.__code__.co_varnames:
                # Add both normal and inverted operation (if available)
                operations_two.append(op(invert=False))
                operations_two.append(op(invert=True))
            else:
                operations_two.append(op())

            active_operations.append(aliases_two[name])
        else:
            print(f"Operation {name} not found!")

    return operations_one, operations_two


class Solver:
    """
    Solver to perform operations on numbers to see if target number(s) can be reached.

    Inputs
    ------
    operations: list[str]
        ...
    reduce_multiple_answers: bool
        ...

    Returns
    -------
    results: dict[int, np.ndarray[str]]
        ...

    Examples
    --------
    ...
    """
    def __init__(self, operations: list[str], reduce_multiple_answers: bool = False):
        self._operations = operations
        self.reduce_multiple_answers = reduce_multiple_answers

        self.sskg = StirlingSecondKindGenerator()

        # Operations with one or two parameters
        self.operations_one = None
        self.operations_two = None

        # Parameters concerning inputs
        self.inputs = None
        self.n = None
        self.range_n = None
        self.duplicates = None

        # Parameters for tracking the results
        self.results = None
        self.results_string = None

    def __call__(self, inputs: list[int, float], targets: Union[int, float, list[int, float]]) -> dict[Union[int, float], np.ndarray[str]]:
        # Initialize the result variables
        self.initialize(inputs)

        # Calculate all combinations
        self.calculate_combinations()

        # Get computations for targets
        return self.get_targets_computation(targets)

    def initialize(self, inputs):
        """ Initialize parameters given the input numbers. """
        assert inputs is not None, "`inputs` cannot be None!"

        # Get list of inputs
        self.inputs = inputs

        if len(np.unique(inputs)) == len(inputs):
            self.duplicates = False
        else:
            self.duplicates = True

        self.n = len(self.inputs)
        self.range_n = range(self.n)

        # Check operations
        if isinstance(self._operations, list) and isinstance(self._operations[0], str):
            self.operations_one, self.operations_two = get_operations(self._operations)

        # Initialize the results dictionary
        # TODO explain structure
        self.results = {x + 1: dict() for x in self.range_n}
        self.results[1] = {((i, ), ): np.array([self.inputs[i]]) for i in self.range_n}

        # Initialize the results string dictionary
        # Structure is similar to the `results`.
        self.results_string = {x + 1: dict() for x in self.range_n}
        self.results_string[1] = {((i,),): np.array([str(self.inputs[i])]) for i in self.range_n}

    def _remove_duplicates(self, results, results_string):
        """ Remove duplicates (if required) """
        if self.duplicates and len(results_string) > 1:
            unique_str, idx = np.unique(results_string, return_index=True)
            if len(unique_str) < len(results_string):
                results = results[idx]
                results_string = unique_str

        return results, results_string

    def _reduce_multiple_answers(self, results, results_string):
        """ Reduce multiple answers (if required)"""
        if self.reduce_multiple_answers and len(results) > 1:
            # Keep only the first possibility per unique answer
            unique, idx = np.unique(results, return_index=True)
            if len(unique) < len(results):
                results = unique
                results_string = results_string[idx]

        return results, results_string

    def calculate_combinations(self):
        """
        Calculate the combinations that can be made from the input numbers. The results are tracked by
        `results` and the calculation is tracked by `results_string`.
        """
        for _n in range(2, len(self.inputs) + 1):

            sskg_combinations = self.sskg(_n, 2)
            combis = list(combinations([(x, ) for x in self.range_n], _n))
            # print(combis)
            for combi in combis:
                # print(combi)
                self.results[_n][combi] = np.array([])
                self.results_string[_n][combi] = np.array([], dtype='<U7')
                for sskg_combi in sskg_combinations:
                    idx_part1, idx_part2 = sskg_combi
                    part1 = tuple(combi[idx] for idx in idx_part1)
                    part2 = tuple(combi[idx] for idx in idx_part2)

                    # Find the possible numbers for the first and second part to be fed into the operation
                    p1 = self.results[len(part1)][part1]
                    p2 = self.results[len(part2)][part2]

                    # Find the possible formulas for the first and second part to be fed into the string operation
                    p1_string = self.results_string[len(part1)][part1]
                    p2_string = self.results_string[len(part2)][part2]

                    if len(self.operations_two) > 0:
                        # Loop over all active two parameter operations and append the results
                        for operation in self.operations_two:
                            result, result_str = operation(p1, p2, p1_string, p2_string)
                            self.results[_n][combi] = np.append(self.results[_n][combi], result)
                            self.results_string[_n][combi] = np.append(self.results_string[_n][combi], result_str)

                        # When working with duplicate input numbers, there might be duplicates in the calculations.
                        # Remove these.
                        self.results[_n][combi], self.results_string[_n][combi] = self._remove_duplicates(self.results[_n][combi], self.results_string[_n][combi])

                        # Keep only the first possibility per unique answer
                        self.results[_n][combi], self.results_string[_n][combi] = self._reduce_multiple_answers(self.results[_n][combi], self.results_string[_n][combi])

                    if len(self.operations_one) > 0:
                        # Loop over all active one parameter operations and append the results
                        for operation in self.operations_one:
                            result, result_str = operation(self.results[_n][combi], self.results_string[_n][combi])
                            self.results[_n][combi] = np.append(self.results[_n][combi], result)
                            self.results_string[_n][combi] = np.append(self.results_string[_n][combi], result_str)

                        # When working with duplicate input numbers, there might be duplicates in the calculations.
                        # Remove these.
                        self.results[_n][combi], self.results_string[_n][combi] = self._remove_duplicates(self.results[_n][combi], self.results_string[_n][combi])

                        # Keep only the first possibility per unique answer
                        self.results[_n][combi], self.results_string[_n][combi] = self._reduce_multiple_answers(self.results[_n][combi], self.results_string[_n][combi])

    def get_targets_computation(self, targets: Union[int, float, list[int, float]]) -> dict[Union[int, float], np.ndarray[str]]:
        """
        For a single target or list of targets, get the computation to achieve the target.
        """
        # Get the results at the highest level (i.e. self.n)
        results = list(self.results[self.n].values())[0]
        results_string = list(self.results_string[self.n].values())[0]

        # Check if the target(s) have been found
        if isinstance(targets, (int, float)):
            targets = [targets]

        out = dict()
        for target in targets:
            isin = np.where(results == target)[0]
            if len(isin) > 0:
                out[target] = results_string[isin]
            else:
                out[target] = "Not possible!"

        return out


class StirlingSecondKindGenerator:
    """
    Generates all combinations of n partition into k subsets. In other words:
    "How many ways can you place n marked balls into k plain boxes, with no empty boxes allowed?"
    For more information see https://en.wikipedia.org/wiki/Twelvefold_way

    Example
    -------
    >>> StirlingSecondKindGenerator()(n=4, k=2)
    [
        ((0, 1, 2), (3,)),
        ((0, 1, 3), (2,)),
        ((0, 1), (2, 3)),
        ((0, 2, 3), (1,)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
        ((0,), (1, 2, 3))
    ]
    """
    def __init__(self):
        self.results = []

    def __call__(self, n, k, items=None):
        self.results = []
        parts = [[] for _ in range(k)]
        self.generate_parts(parts, empty=k, n=n, k=k, m=0, last_filled=-1)

        if items is None:
            return self.results
        else:
            return [
                [[items[idx] for idx in index_lists] for index_lists in result] for result in self.results
            ]

    def generate_parts(self, parts: list, empty, n, k, m, last_filled):
        """ Loops iterative over the numbers to add the possibilities. """
        if m == n:
            self.results.append(
                tuple(tuple(_list) for _list in parts)
            )
            return

        if n - m == empty:
            start = k - empty
        else:
            start = 0

        for i in range(start, min(k, last_filled + 2)):
            parts[i].append(m)
            if len(parts[i]) == 1:
                empty -= 1
            self.generate_parts(parts, empty, n, k, m + 1, max(i, last_filled))
            parts[i].pop()
            if len(parts[i]) == 0:
                empty += 1
