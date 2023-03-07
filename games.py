from solver import Solver, get_operations


class TwentyFourGame:
    """
    Twenty four (24) game

    Given four numbers between 0 and 9, try to make 24. Allowed operations are addition, subtraction,
    multiplication and division. For example:
    1, 3, 5, 7 --> (5 + 1) * (7 - 3) = 24

    Examples
    --------
    >>> TwentyFourGame()([1, 3, 5, 7])
    array(['((3 - 1) * (5 + 7))', '((1 + 5) * (7 - 3))'], dtype='<U19')
    """
    def __init__(self):
        operation_names = ['plus', 'min', 'times', 'divide']
        operations = get_operations(operation_names)
        self.s = Solver(operations)

    def __call__(self, numbers):
        return self.s(numbers, 24)[24]


class SevenSevensGame:
    """
    Seven sevens game

    Given seven sevens, try to make as many natural numbers as possible, starting at zero.
    Allowed operations are addition, subtraction, multiplication, division, powers and TODO .

    0 = (7 ** (7 / 7)) / 7 - 7 / 7 + 7 - 7
    1 = (7 ** (7 / 7)) / 7 - 7 / 7 + 7 / 7
    2 = (7 ** (7 / 7)) / 7 + 7 / 7 + 7 - 7
    ...

    Examples
    --------
    >>> SevenSevensGame()(range(3))
    {
        0: array(['((((((7 * 7) * 7) / 7) / 7) - 7) * 7)']),
        1: array(['((((((7 * 7) * 7) / 7) + 7) / 7) - 7)']),
        2: array(['((((((7 * 7) * 7) / 7) / 7) + 7) / 7)'])
    }
    """

    def __init__(self):
        operation_names = ['plus', 'min', 'times', 'divide', 'power']
        operations = get_operations(operation_names)
        self.s = Solver(operations, reduce_multiple_answers=True)

        # Check for calculation
        self._did_calc = False

    def calc(self):
        """ Pre-calculate all possibilities. This takes a minute or two... """
        numbers = [7] * 7
        self.s.initialize(numbers)
        print("Calculating...")
        self.s.calculate_combinations()
        print("Done!")
        self._did_calc = True

    def __call__(self, targets):
        if not self._did_calc:
            self.calc()

        return self.s.get_targets_computation(targets)


if __name__ == "__main__":
    a = SevenSevensGame()(range(100))
    print(a)
