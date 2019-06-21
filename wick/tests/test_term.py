import unittest

from wick.expression import *
from wick.orbital_space import OrbitalSpace

class TermTest(unittest.TestCase):
    def test_scalar_mul(self):
        o = OrbitalSpace("o")
        s = 1.0
        sums = [Sigma("i", o), Sigma("j", o)]
        tensors = [Tensor(["i","j"], [o,o], 'f')]
        operators = [Operator("i", o, True), Operator("j", o, False)]
        t = Term(s, sums, tensors, operators, [])
        t1 = 3.14*t
        t2 = t*3.14
        self.assertTrue(t1 == t2)

    def test_mul(self):
        o = OrbitalSpace("o")
        v = OrbitalSpace("v")
        sum1 = [Sigma("i", o), Sigma("j", o)]
        ten1 = [Tensor(["i","j"], [o,o], 'f')]
        ops1 = [Operator("i", o, True), Operator("j", o, False)]
        t1 = Term(1.0, sum1, ten1, ops1, [])
        sum2 = [Sigma("a", v), Sigma("b", v)]
        ten2 = [Tensor(["a","b"], [v,v], 'f')]
        ops2 = [Operator("a", v, True), Operator("b", v, False)]
        t2 = Term(1.0, sum2, ten2, ops2, [])

        sum3 = sum2 + sum1
        ten3 = ten1 + ten2
        ops3 = ops1 + ops2 
        ref = Term(1.0, sum3, ten3, ops3, [])

        out = t1*t2
        self.assertTrue(ref == out)

if __name__ == '__main__':
    unittest.main()
