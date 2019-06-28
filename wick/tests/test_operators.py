import unittest

from wick.index import Idx
from wick.operator import *

class OperatorTest(unittest.TestCase):
    def test_operator(self):
        i = Idx(0,"occ")
        j = Idx(1,"occ")
        O1 = Operator(i, False)
        O2 = Operator(i, False)
        O3 = Operator(j, False)
        O4 = Operator(j, True)
        self.assertTrue(O1 == O2)
        self.assertTrue(O1 != O3)
        self.assertTrue(O1 != O4)
        self.assertTrue(O2 != O4)

    def test_tensor(self):
        i = Idx(0,"occ")
        a = Idx(0,"vir")
        T1 = Tensor([i,a], "g")
        T2 = Tensor([i,a], "f")
        T3 = Tensor([i,a], "f")
        T4 = Tensor([a,i], "f")
        self.assertTrue(T2 == T3)
        self.assertTrue(T2 != T4)
        self.assertTrue(T1 != T3) 
        self.assertTrue(T1 != T4) 

    def test_sigma(self):
        i = Idx(0,"occ")
        j = Idx(1,"occ")
        a = Idx(0,"vir")
        S1 = Sigma(i)
        S2 = Sigma(i)
        S3 = Sigma(j)
        S4 = Sigma(a)
        self.assertTrue(S2 == S2)
        self.assertTrue(S2 != S3)
        self.assertTrue(S2 != S4)
        self.assertTrue(S3 != S4)

    def test_delta(self):
        i = Idx(0,"occ")
        j = Idx(1,"occ")
        a = Idx(0,"vir")
        b = Idx(1,"vir")
        D1 = Delta(i,j)
        D2 = Delta(i,j)
        D3 = Delta(j,i)
        D4 = Delta(a,b)
        self.assertTrue(D1 == D2)
        self.assertTrue(D1 == D3)
        self.assertTrue(D1 != D4)

if __name__ == '__main__':
    unittest.main()
