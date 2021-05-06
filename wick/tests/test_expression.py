import unittest

from wick.index import Idx
from wick.operator import FOperator, Sigma, Tensor, Delta
from wick.expression import Term


class ExpressionTest(unittest.TestCase):
    def test_resolve0(self):
        I1 = Idx(0, "o1")
        I2 = Idx(1, "o1")
        I3 = Idx(0, "o2")
        operators = [FOperator(I1, True), FOperator(I3, False)]
        out = Term(
            1, [Sigma(I3)],
            [Tensor([I1, I3], "T")],
            operators, [Delta(I1, I2)])

        out.resolve()

        ref = Term(
            1, [Sigma(I3)],
            [Tensor([I1, I3], "T")],
            operators, [Delta(I1, I2)])

        self.assertTrue(ref == out)

    def test_resolve1(self):
        I1 = Idx(0, "o1")
        I2 = Idx(1, "o1")
        I3 = Idx(0, "o2")
        operators = [FOperator(I1, True), FOperator(I3, False)]
        out = Term(
            1, [Sigma(I1), Sigma(I3)],
            [Tensor([I1, I3], "T")],
            operators, [Delta(I1, I2)])

        out.resolve()

        roperators = [FOperator(I2, True), FOperator(I3, False)]
        ref = Term(
            1, [Sigma(I3)],
            [Tensor([I2, I3], "T")],
            roperators, [])

        self.assertTrue(ref == out)

    def test_resolve2(self):
        I1 = Idx(0, "o1")
        I3 = Idx(0, "o2")
        I4 = Idx(1, "o2")
        operators = [FOperator(I1, True), FOperator(I3, False)]
        out = Term(
            1, [Sigma(I1), Sigma(I3)],
            [Tensor([I1, I3], "T")],
            operators, [Delta(I4, I3)])

        out.resolve()

        roperators = [FOperator(I1, True), FOperator(I4, False)]
        ref = Term(
            1, [Sigma(I1)],
            [Tensor([I1, I4], "T")],
            roperators, [])

        self.assertTrue(ref == out)

    def test_resolve3(self):
        I1 = Idx(0, "o1")
        I2 = Idx(1, "o1")
        operators = [FOperator(I1, True), FOperator(I2, False)]
        out = Term(
            1, [Sigma(I1), Sigma(I2)],
            [Tensor([I1, I2], "T")],
            operators, [Delta(I1, I2)])
        out.resolve()

        roperators = [FOperator(I1, True), FOperator(I1, False)]
        ref = Term(
            1, [Sigma(I1)],
            [Tensor([I1, I1], "T")],
            roperators, [])

        self.assertTrue(ref == out)

    def test_resolve_chain(self):
        I1 = Idx(0, "v1")
        I2 = Idx(0, "o1")
        I3 = Idx(1, "v1")
        I4 = Idx(1, "o1")
        operators = [
            FOperator(I1, True), FOperator(I2, True),
            FOperator(I4, False), FOperator(I3, False)]
        out = Term(
            1, [Sigma(I2), Sigma(I3), Sigma(I4)],
            [Tensor([I1, I2, I3, I4], "T")],
            operators, [Delta(I1, I3), Delta(I2, I4)])
        out.resolve()

        roperators = [
            FOperator(I1, True), FOperator(I2, True),
            FOperator(I2, False), FOperator(I1, False)]
        ref = Term(
            1, [Sigma(I2)],
            [Tensor([I1, I2, I1, I2], "T")],
            roperators, [])

        self.assertTrue(ref == out)


if __name__ == '__main__':
    unittest.main()
