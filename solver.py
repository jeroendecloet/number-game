from inspect import isclass
from typing import Union

import numpy as np
from itertools import combinations

import operations as ops


def get_operations(operation_names: list[str]) -> list[callable]:
    """
    Translates a list of operation names to actual instanced operations.

    Inputs
    ------
    operation_names: list
        List of names of operations to use, e.g. `plus` or `multiplication`

    Outputs
    -------
    operations: list
        List with instanced/callable operations

    """
    all_operation_names = [_class for _class in dir(ops) if (isclass(getattr(ops, _class))) and (_class != 'BaseOperation') and issubclass(getattr(ops, _class), ops.BaseOperation)]

    # Make a list of all aliases of the operations
    aliases = dict()
    for name in all_operation_names:
        op = getattr(ops, name)
        aliases = {**aliases, **dict(zip(op.names, [name] * len(op.names)))}

    # Look through the operations to
    operations = list()
    active_operations = list()
    for name in operation_names:
        # I
        if name in aliases:
            # Check if the operation has not been added already
            if aliases[name] in active_operations:
                print(f"Operation {name} already found! Skipping...")
                continue

            op = getattr(ops, aliases[name])
            # Initialize operations
            if 'invert' in op.__init__.__code__.co_varnames:
                # Add both normal and inverted operation (if available)
                operations.append(op(invert=False))
                operations.append(op(invert=True))
            else:
                operations.append(op())

            active_operations.append(aliases[name])
        else:
            print(f"Operation {name} not found!")

    return operations


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
    def __init__(self, operations: list[Union[str, callable]], reduce_multiple_answers: bool = False):
        self.operations = operations
        self.reduce_multiple_answers = reduce_multiple_answers

        self.sskg = StirlingSecondKindGenerator()

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
        if isinstance(self.operations, list) and isinstance(self.operations[0], str):
            self.operations = get_operations(self.operations)

        # Initialize the results dictionary
        # TODO explain structure
        self.results = {x + 1: dict() for x in self.range_n}
        self.results[1] = {((i, ), ): np.array([self.inputs[i]]) for i in self.range_n}

        # Initialize the results string dictionary
        # Structure is similar to the `results`.
        self.results_string = {x + 1: dict() for x in self.range_n}
        self.results_string[1] = {((i,),): np.array([str(self.inputs[i])]) for i in self.range_n}

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

                    # Loop over all active operations and append the results
                    for operation in self.operations:
                        result, result_str = operation(p1, p2, p1_string, p2_string)
                        self.results[_n][combi] = np.append(self.results[_n][combi], result)
                        self.results_string[_n][combi] = np.append(self.results_string[_n][combi], result_str)

                    # When working with duplicate input numbers, there might be duplicates in the calculations.
                    # Remove these.
                    if self.duplicates and len(self.results_string[_n][combi]) > 1:
                        unique_str, idx = np.unique(self.results_string[_n][combi], return_index=True)
                        if len(unique_str) < len(self.results_string[_n][combi]):
                            self.results[_n][combi] = self.results[_n][combi][idx]
                            self.results_string[_n][combi] = unique_str

                    if self.reduce_multiple_answers and len(self.results[_n][combi]) > 1:
                        # Keep only the first possibility per unique answer
                        unique, idx = np.unique(self.results[_n][combi], return_index=True)
                        if len(unique) < len(self.results[_n][combi]):
                            self.results[_n][combi] = unique
                            self.results_string[_n][combi] = self.results_string[_n][combi][idx]

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


if __name__ == "__main__":
    # sskg = StirlingSecondKindGenerator()
    # r = sskg(4, 2)
    # print(r)
    # print(len(r))

    operations = get_operations(['plus', 'min', 'times', 'division'])

    solver = Solver(operations=operations)

    numbers = [3, 3, 7, 7]
    out = solver(numbers, 24)
    print(out)
