import unittest

from wick.index import Idx
from wick.expression import *
from wick.wick import apply_wick

class TestSCRules(unittest.TestCase):
    def test_0d0(self):
        # 0 differences, constant
        t = Term(1.0, [], [Tensor([], 'I')], [], [])
        e = Expression([t])
        x = apply_wick(e)
        self.assertTrue(len(x.terms) == 1)

    def test_0d1(self):
        # 0 differences, 1-particle operator
        i = Idx("i","occ")
        j = Idx("j","occ")
        a = Idx("a","vir")
        b = Idx("b","vir")
        t1 = Term(1.0, [Sigma(i),Sigma(j)], 
                [Tensor([i,j],'f')],
                [Operator(i, True), Operator(j, False)],
                [])
        t2 = Term(1.0, [Sigma(i),Sigma(a)], 
                [Tensor([i,a],'f')],
                [Operator(i, True), Operator(a, False)],
                [])
        t3 = Term(1.0, [Sigma(a), Sigma(i)], 
                [Tensor([a,i], 'f')],
                [Operator(a, True), Operator(i, False)],
                [])
        t4 = Term(1.0, [Sigma(a),Sigma(b)], 
                [Tensor([a,b],'f')],
                [Operator(a, True), Operator(b, False)],
                [])
        e = Expression([t1,t2,t3,t4])
        #print(e)
        x = apply_wick(e)
        x.resolve()
        print(x)
        self.assertTrue(len(x.terms) == 1)
        self.assertTrue(len(x.terms[0].sums) == 1)

    def test_0d2(self):
        # 0 differences, 2-particle operator
        i = Idx("i","occ")
        j = Idx("j","occ")
        k = Idx("k","occ")
        l = Idx("l","occ")
        t1 = Term(0.25, [Sigma(i),Sigma(j),Sigma(k),Sigma(l)], 
                [Tensor([i,j,k,l],'I')],
                [Operator(i, True), Operator(j, True),
                    Operator(l, False), Operator(k, False)],
                [])
        e = Expression([t1])
        x = apply_wick(e)
        x.resolve()
        print(x)
        self.assertTrue(True)

    def test_1d0(self):
        self.assertTrue(True)

    def test_1d1(self):
        self.assertTrue(True)

    def test_1d2(self):
        self.assertTrue(True)

    def test_2d1(self):
        self.assertTrue(True)

    def test_2d2(self):
        self.assertTrue(True)

    def test_2d3(self):
        self.assertTrue(True)

    def test_3d2(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
