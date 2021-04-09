import unittest
from fractions import Fraction

from wick.index import Idx
from wick.operator import *
from wick.expression import *
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, get_sym

class SCRulesTest(unittest.TestCase):
    def test_0d0(self):
        # 0 differences, constant
        t = Term(1.0, [], [Tensor([], 'I')], [], [])
        e = Expression([t])
        x = apply_wick(e)
        self.assertTrue(x == e)

    def test_0d1(self):
        # 0 differences 1-electron operator
        e = one_e("f", ["occ","vir"])
        x = apply_wick(e)
        x.resolve()
        out = AExpression(Ex=x)
        i = Idx(0, "occ")
        tr1 = ATerm(scalar=1, sums=[Sigma(i)],
                tensors=[Tensor([i,i], "f")])
        ref = AExpression(terms=[tr1])
        self.assertTrue(ref.pmatch(out))

    def test_0d2(self):
        # 0 differences 2-electron operator
        e = two_e("I", ["occ","vir"])
        e = Expression(e.terms[0:1])
        x = apply_wick(e)
        x.resolve()
        out = AExpression(Ex=x)

        i = Idx(0, "occ")
        j = Idx(1, "occ")
        tr1 = ATerm(scalar=Fraction(1,2), sums=[Sigma(i), Sigma(j)],
                tensors=[Tensor([i,j,i,j], "I", sym=get_sym(True))])
        ref = AExpression(terms=[tr1]) 
        self.assertTrue(ref.pmatch(out))

    #def test_1d0(self):
    #    self.assertTrue(True)

    #def test_1d1(self):
    #    self.assertTrue(True)

    #def test_1d2(self):
    #    self.assertTrue(True)

    #def test_2d1(self):
    #    self.assertTrue(True)

    #def test_2d2(self):
    #    self.assertTrue(True)

    #def test_2d3(self):
    #    self.assertTrue(True)

    #def test_3d2(self):
    #    self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
