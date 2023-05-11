import unittest

from number_game.solver import StirlingSecondKindGenerator


class TestStirlingSecondKindGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = StirlingSecondKindGenerator()

    def test_call(self):
        test_cases = [
            {'n': 4, 'k': 2, 'items': None, 'expected': [
                ((0, 1, 2), (3,)),
                ((0, 1, 3), (2,)),
                ((0, 1), (2, 3)),
                ((0, 2, 3), (1,)),
                ((0, 2), (1, 3)),
                ((0, 3), (1, 2)),
                ((0,), (1, 2, 3))
            ]},
            {'n': 3, 'k': 1, 'items': None, 'expected': [
                ((0, 1, 2),),
            ]},
            {'n': 5, 'k': 3, 'items': None, 'expected': [
                ((0, 1, 2), (3,), (4,)),
                ((0, 1, 3), (2,), (4,)),
                ((0, 1), (2, 3), (4,)),
                ((0, 1, 4), (2,), (3,)),
                ((0, 1), (2, 4), (3,)),
                ((0, 1), (2,), (3, 4)),
                ((0, 2, 3), (1,), (4,)),
                ((0, 2), (1, 3), (4,)),
                ((0, 2, 4), (1,), (3,)),
                ((0, 2), (1, 4), (3,)),
                ((0, 2), (1,), (3, 4)),
                ((0, 3), (1, 2), (4,)),
                ((0,), (1, 2, 3), (4,)),
                ((0, 4), (1, 2), (3,)),
                ((0,), (1, 2, 4), (3,)),
                ((0,), (1, 2), (3, 4)),
                ((0, 3, 4), (1,), (2,)),
                ((0, 3), (1, 4), (2,)),
                ((0, 3), (1,), (2, 4)),
                ((0, 4), (1, 3), (2,)),
                ((0,), (1, 3, 4), (2,)),
                ((0,), (1, 3), (2, 4)),
                ((0, 4), (1,), (2, 3)),
                ((0,), (1, 4), (2, 3)),
                ((0,), (1,), (2, 3, 4))
            ]},
            {'n': 3, 'k': 2, 'items': ['a', 'b', 'c'], 'expected': [
                (('a', 'b'), ('c',)),
                (('a', 'c'), ('b',)),
                (('a',), ('b', 'c')),
            ]}
        ]

        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                expected = test_case.pop("expected")
                self.assertEqual(self.generator(**test_case), expected)


if __name__ == '__main__':
    unittest.main()
