import unittest

from wick.index import Idx
from wick.expression import Term, Expression, AExpression
from wick.operator import FOperator, BOperator, Projector
from wick.wick import valid_contraction, pair_list
from wick.wick import get_sign, split_operators, apply_wick
from wick.convenience import one_e


class WickTest(unittest.TestCase):
    def test_valid_contraction(self):
        i = Idx(0, "occ")
        j = Idx(1, "occ")
        a = Idx(0, "vir")
        b = Idx(1, "vir")
        O1 = FOperator(i, False)
        O2 = FOperator(j, True)
        O3 = FOperator(a, False)
        O4 = FOperator(b, True)
        self.assertTrue(valid_contraction(O2, O1))
        self.assertTrue(not valid_contraction(O1, O2))
        self.assertTrue(valid_contraction(O3, O4))
        self.assertTrue(not valid_contraction(O4, O3))
        self.assertTrue(not valid_contraction(O1, O3))

        x = Idx(0, "nm", fermion=False)
        y = Idx(1, "nm", fermion=False)
        Ob1 = BOperator(x, False)
        Ob2 = BOperator(y, True)
        self.assertTrue(not valid_contraction(Ob2, Ob1))
        self.assertTrue(valid_contraction(Ob1, Ob2))
        self.assertTrue(not valid_contraction(Ob2, O1))

    def test_pair_list(self):
        i = Idx(0, "occ")
        j = Idx(1, "occ")
        k = Idx(2, "occ")
        l = Idx(3, "occ")
        O1 = FOperator(i, False)
        O2 = FOperator(j, True)
        O3 = FOperator(k, False)
        O4 = FOperator(l, True)

        os = [O2, O4, O1, O3]
        pl = pair_list(os)
        self.assertTrue(len(pl) == 2)

        os = [O2, O1, O4, O3]
        pl = pair_list(os)
        self.assertTrue(len(pl) == 1)

    def test_get_sign(self):
        ipairs = [(0, 1), (2, 3)]
        self.assertTrue(get_sign(ipairs) == 1)

        ipairs = [(0, 2), (1, 3)]
        self.assertTrue(get_sign(ipairs) == -1)

    def test_split_operators(self):
        i = Idx(0, "occ")
        j = Idx(1, "occ")
        k = Idx(3, "occ")
        l = Idx(4, "occ")
        O1 = FOperator(i, False)
        O2 = FOperator(j, False)
        O3 = FOperator(k, False)
        O4 = FOperator(l, True)
        P = Projector()

        ops = [O1, O2, P, O3, P, O4]
        olists = split_operators(ops)
        self.assertTrue(len(olists) == 3)
        self.assertTrue(olists[0] == [O1, O2])
        self.assertTrue(olists[1] == [O3])
        self.assertTrue(olists[2] == [O4])

    def test_projector(self):
        O1 = one_e("f", ["occ", "vir"])
        O2 = one_e("g", ["occ", "vir"])
        ref1 = apply_wick(O1)
        ref2 = apply_wick(O2)
        ref1.resolve()
        ref2.resolve()
        ref = AExpression(Ex=ref1*ref2)

        P = Expression([Term(1, [], [], [Projector()], [])])
        out1 = apply_wick(O1*P*O2)
        out1.resolve()
        out = AExpression(Ex=out1)
        self.assertTrue(ref.pmatch(out))


if __name__ == '__main__':
    unittest.main()
