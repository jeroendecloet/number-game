from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from number_game import operations as ops


class TestOperations(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.operation_tests = {
            "Addition": {
                "in": (np.array([0, 1]), np.array([1, 2])),
                "out": np.array([1, 2, 2, 3]),
                "out_str": np.array(["(0 + 1)", "(1 + 1)", "(0 + 2)", "(1 + 2)"])
            },
            "Subtraction": {
                "in": (np.array([0, 1]), np.array([1, 2])),
                "out": np.array([-1, 0, -2, -1]),
                "out_str": np.array(["(0 - 1)", "(1 - 1)", "(0 - 2)", "(1 - 2)"]),
                "out_invert": np.array([1, 0, 2, 1]),
                "out_str_invert": np.array(["(1 - 0)", "(1 - 1)", "(2 - 0)", "(2 - 1)"]),
            },
            "Multiplication": {
                "in": (np.array([0, 1]), np.array([2, 3])),
                "out": np.array([0, 2, 0, 3]),
                "out_str": np.array(["(0 * 2)", "(1 * 2)", "(0 * 3)", "(1 * 3)"])
            },
            "Division": {
                "in": (np.array([0, 1]), np.array([1, 2])),
                "out": np.array([0, 1, 0, 0.5]),
                "out_str": np.array(["(0 / 1)", "(1 / 1)", "(0 / 2)", "(1 / 2)"]),
                "out_invert": np.array([1, 2]),
                "out_str_invert": np.array(["(1 / 1)", "(2 / 1)"]),
            }
        }
        for args in cls.operation_tests.values():
            args['in_str'] = (args['in'][0].astype(str), args['in'][1].astype(str))

    def test_operation(self):
        """ Tests the operation """
        for op, args in self.operation_tests.items():
            with self.subTest(f"Operation {op}", op=op):

                if "out_invert" in args:
                    for invert in [True, False]:
                        with self.subTest(f"Invert {invert}", invert=invert):
                            op_instance = getattr(ops, op)(invert=invert)
                            result, result_str = op_instance(*args['in'], *args['in_str'])
                            assert_allclose(
                                result,
                                args['out_invert'] if invert else args['out']
                            )
                            assert_equal(
                                result_str,
                                args['out_str_invert'] if invert else args['out_str']
                            )
                else:
                    op_instance = getattr(ops, op)()
                    result, result_str = op_instance(*args['in'], *args['in_str'])
                    assert_allclose(
                        result,
                        args['out']
                    )
                    assert_equal(
                        result_str,
                        args['out_str']
                    )
