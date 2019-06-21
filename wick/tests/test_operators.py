import unittest

from wick.orbital_space import OrbitalSpace
from wick.operator import *

class OperatorTest(unittest.TestCase):
    def test_operator(self):
        o = OrbitalSpace("o")
        O1 = Operator("i", o, False)
        O2 = Operator("i", o, False)
        O3 = Operator("j", o, False)
        O4 = Operator("j", o, True)
        self.assertTrue(O1 == O2)
        self.assertTrue(O1 != O3)
        self.assertTrue(O1 != O4)
        self.assertTrue(O2 != O4)

    def test_tensor(self):
        o = OrbitalSpace("o")
        v = OrbitalSpace("v")
        T1 = Tensor(["i","a"], [o,v], "g")
        T2 = Tensor(["i","a"], [o,v], "f")
        T3 = Tensor(["i","a"], [o,v], "f")
        T4 = Tensor(["a","i"], [v,o], "f")
        self.assertTrue(T2 == T3)
        self.assertTrue(T2 != T4)
        self.assertTrue(T1 != T3) 
        self.assertTrue(T1 != T4) 

    def test_sigma(self):
        o = OrbitalSpace("o")
        v = OrbitalSpace("v")
        S1 = Sigma("i",o)
        S2 = Sigma("i",o)
        S3 = Sigma("j",o)
        S4 = Sigma("a",v)
        self.assertTrue(S2 == S2)
        self.assertTrue(S2 != S3)
        self.assertTrue(S2 != S4)
        self.assertTrue(S3 != S4)

    def test_delta(self):
        o = OrbitalSpace("o")
        v = OrbitalSpace("v")
        D1 = Delta("i","j",o,o)
        D2 = Delta("i","j",o,o)
        D3 = Delta("j","i",o,o)
        D4 = Delta("a","b",v,v)
        self.assertTrue(D1 == D2)
        self.assertTrue(D1 == D3)
        self.assertTrue(D1 != D4)

if __name__ == '__main__':
    unittest.main()
