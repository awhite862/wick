import unittest

from wick.index import Idx
from wick.expression import *

class TermTest(unittest.TestCase):
    def test_scalar_mul(self):
        s = 1.0
        i = Idx("i","occ")
        j = Idx("j","occ")
        sums = [Sigma(i), Sigma(j)]
        tensors = [Tensor([i,j], 'f')]
        operators = [Operator(i, True), Operator(j, False)]
        t = Term(s, sums, tensors, operators, [])
        t1 = 3.14*t
        t2 = t*3.14
        self.assertTrue(t1 == t2)

    def test_mul(self):
        i = Idx("i","occ")
        j = Idx("j","occ")
        a = Idx("a","vir")
        b = Idx("b","vir")
        sum1 = [Sigma(i), Sigma(j)]
        ten1 = [Tensor([i,j], 'f')]
        ops1 = [Operator(i, True), Operator(j, False)]
        t1 = Term(1.0, sum1, ten1, ops1, [])
        sum2 = [Sigma(a), Sigma(b)]
        ten2 = [Tensor([a,b], 'f')]
        ops2 = [Operator(a, True), Operator(b, False)]
        t2 = Term(1.0, sum2, ten2, ops2, [])

        sum3 = sum2 + sum1
        ten3 = ten1 + ten2
        ops3 = ops1 + ops2 
        ref = Term(1.0, sum3, ten3, ops3, [])

        out = t1*t2
        self.assertTrue(ref == out)

if __name__ == '__main__':
    unittest.main()
