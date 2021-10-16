import unittest
from wick.operator import Tensor, Sigma
from wick.expression import ATerm
from wick.index import Idx


class TermMapTest(unittest.TestCase):
    def test_null(self):
        i = Idx(0, "occ")
        j = Idx(1, "occ")
        a = Idx(0, "vir")
        b = Idx(1, "vir")
        L1 = Tensor([i, j, a, b], "L")
        J1 = Tensor([j, b], "J")
        S1j = Sigma(j)
        S1b = Sigma(b)
        T1 = ATerm(
            scalar=1,
            sums=[S1j, S1b],
            tensors=[L1, J1])
        T2 = ATerm(
            scalar=1,
            sums=[S1j, S1b],
            tensors=[L1, J1])
        self.assertTrue(T1.match(T2))
        self.assertTrue(T2.match(T1))

    def test_label(self):
        i = Idx(0, "occ")
        j = Idx(1, "occ")
        a = Idx(0, "vir")
        b = Idx(1, "vir")
        L1 = Tensor([i, j, a, b], "L")
        J1 = Tensor([j, b], "J")
        S1j = Sigma(j)
        S1a = Sigma(a)
        S1b = Sigma(b)
        T1 = ATerm(
            scalar=1,
            sums=[S1j, S1b],
            tensors=[L1, J1])
        L2 = Tensor([i, j, b, a], "L")
        J2 = Tensor([j, a], "J")
        T2 = ATerm(
            scalar=1,
            sums=[S1j, S1a],
            tensors=[L2, J2])
        self.assertTrue(T1.match(T2))
        self.assertTrue(T2.match(T1))


if __name__ == '__main__':
    unittest.main()
