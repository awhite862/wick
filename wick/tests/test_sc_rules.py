import unittest

from wick.index import Idx
from wick.operator import *
from wick.expression import *
from wick.wick import apply_wick
from wick.hamiltonian import one_e, two_e

class TestSCRules(unittest.TestCase):
    def test_0d0(self):
        # 0 differences, constant
        t = Term(1.0, [], [Tensor([], 'I')], [], [])
        e = Expression([t])
        x = apply_wick(e)
        self.assertTrue(len(x.terms) == 1)

    def test_0d1(self):
        e = one_e("f", ["occ","vir"])
        x = apply_wick(e)
        x.resolve()
        #print(x)
        self.assertTrue(len(x.terms) == 1)
        self.assertTrue(len(x.terms[0].sums) == 1)

    def test_0d2(self):
        e = two_e("I", ["occ","vir"])
        e = Expression(e.terms[0:1])
        print(e._print_str())
        x = apply_wick(e)
        print(x._print_str())
        x.resolve()
        print(x._print_str())
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
