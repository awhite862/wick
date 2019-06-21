import unittest

from wick.expression import *
from wick.orbital_space import OrbitalSpace
from wick.wick import apply_wick

class TestSCRules(unittest.TestCase):
    def test_0d0(self):
        # 0 differences, constant
        t = Term(1.0, [], [Tensor([],[], 'I')], [], [])
        e = Expression([t])
        x = apply_wick(e)
        self.assertTrue(len(x.terms) == 1)

    def test_0d1(self):
        # 0 differences, 1-particle operator
        o = OrbitalSpace("o")
        v = OrbitalSpace("v")
        t1 = Term(1.0, [Sigma('i',o),Sigma('j',o)], 
                [Tensor(['i','j'],[o,o],'f')],
                [Operator('i', o, True), Operator('j', o, False)],
                [])
        t2 = Term(1.0, [Sigma('i',o),Sigma('a',v)], 
                [Tensor(['i','a'],[o,v],'f')],
                [Operator('i', o, True), Operator('a', v, False)],
                [])
        t3 = Term(1.0, [Sigma('a',v),Sigma('i',o)], 
                [Tensor(['a','i'],[v,o],'f')],
                [Operator('a', v, True), Operator('i', o, False)],
                [])
        t4 = Term(1.0, [Sigma('a',v),Sigma('b',v)], 
                [Tensor(['a','b'],[v,v],'f')],
                [Operator('a', v, True), Operator('b', v, False)],
                [])
        e = Expression([t1,t2,t3,t4])
        #print(e)
        x = apply_wick(e)
        x.resolve()
        #print(x)
        self.assertTrue(len(x.terms) == 1)
        self.assertTrue(len(x.terms[0].sums) == 1)

    def test_0d2(self):
        # 0 differences, 2-particle operator
        o = OrbitalSpace("o")
        v = OrbitalSpace("v")
        t1 = Term(0.25, [Sigma('i',o),Sigma('j',o),Sigma('k',o),Sigma('l',o)], 
                [Tensor(['i','j','k','l'],[o,o,o,o],'I')],
                [Operator('i', o, True), Operator('j', o, True),
                    Operator('l', o, False), Operator('k', o, False)],
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
