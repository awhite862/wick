import os
import unittest
from fractions import Fraction

from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, two_p, commute
from wick.convenience import braE1, E1, E2, braP2, P2


def get_ref(fname):
    froot = os.path.dirname(__file__)
    fpath = os.path.join(froot, "..", "..", "examples", fname)
    with open(fpath) as f:
        ref = f.read()
    return ref


class FullTest(unittest.TestCase):
    def test_ccsd_T1(self):
        H1 = one_e("f", ["occ", "vir"], norder=True)
        H2 = two_e("I", ["occ", "vir"], norder=True)
        H = H1 + H2

        bra = braE1("occ", "vir")
        T1 = E1("t", ["occ"], ["vir"])
        T2 = E2("t", ["occ"], ["vir"])
        T = T1 + T2

        HT = commute(H, T)
        HTT = commute(HT, T)
        HTTT = commute(commute(commute(H2, T1), T1), T1)

        S = bra*(H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT)
        out = apply_wick(S)
        out.resolve()
        final = AExpression(Ex=out)
        out = str(final) + "\n"
        ref = get_ref("ccsd_T1.out")
        self.assertTrue(ref == out)

    def test_p2(self):
        H = two_p("w")
        bra = braP2("nm")
        S2 = P2("S2old", ["nm"])
        HT = commute(H, S2)
        HTT = commute(HT, S2)
        S = bra*(H + HT + 0.5*HTT)
        out = apply_wick(S)
        out.resolve()
        final = AExpression(Ex=out)
        out = str(final) + "\n"
        ref = get_ref("p2_test.out")
        self.assertTrue(ref == out)


if __name__ == '__main__':
    unittest.main()
