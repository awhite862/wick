import unittest

from wick.index import Idx
from wick.expression import Term
from wick.operator import FOperator, Sigma, Tensor


class TermTest(unittest.TestCase):
    def test_scalar_mul(self):
        s = 1.0
        i = Idx(0, "occ")
        j = Idx(1, "occ")
        sums = [Sigma(i), Sigma(j)]
        tensors = [Tensor([i, j], 'f')]
        operators = [FOperator(i, True), FOperator(j, False)]
        t = Term(s, sums, tensors, operators, [])
        t1 = 3.14*t
        t2 = t*3.14
        self.assertTrue(t1 == t2)

    def test_mul(self):
        i = Idx(0, "occ")
        j = Idx(1, "occ")
        a = Idx(0, "vir")
        b = Idx(1, "vir")
        sum1 = [Sigma(i), Sigma(j)]
        ten1 = [Tensor([i, j], 'f')]
        ops1 = [FOperator(i, True), FOperator(j, False)]
        t1 = Term(1.0, sum1, ten1, ops1, [])
        sum2 = [Sigma(a), Sigma(b)]
        ten2 = [Tensor([a, b], 'f')]
        ops2 = [FOperator(a, True), FOperator(b, False)]
        t2 = Term(1.0, sum2, ten2, ops2, [])

        sum3 = sum2 + sum1
        ten3 = ten1 + ten2
        ops3 = ops1 + ops2
        ref = Term(1.0, sum3, ten3, ops3, [])

        out = t1*t2
        self.assertTrue(ref == out)

    def test_ilist(self):
        s = 1.0
        i = Idx("i", "occ")
        j = Idx("j", "occ")
        sums = [Sigma(i), Sigma(j)]
        tensors = [Tensor([i, j], 'f')]
        operators = [FOperator(i, True), FOperator(j, False)]
        t1 = Term(s, sums, tensors, operators, [])
        ilist = t1.ilist()
        iref = [i, j]
        self.assertTrue(set(iref) == set(ilist))

    def test_mul2(self):
        s = 1.0
        i = Idx(0, "occ")
        j = Idx(1, "occ")
        sums = [Sigma(i), Sigma(j)]
        tensors = [Tensor([i, j], 'f')]
        operators = [FOperator(i, True), FOperator(j, False)]
        t1 = Term(s, sums, tensors, operators, [])

        t3 = t1*t1
        k = Idx(2, "occ")
        l = Idx(3, "occ")
        sums = [Sigma(i), Sigma(j), Sigma(k), Sigma(l)]
        tensors = [Tensor([i, j], 'f'), Tensor([k, l], 'f')]
        operators = [FOperator(i, True), FOperator(j, False),
                     FOperator(k, True), FOperator(l, False)]
        ttest = Term(s, sums, tensors, operators, [])
        self.assertTrue(t3 == ttest)


if __name__ == '__main__':
    unittest.main()
