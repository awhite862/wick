import unittest

from wick.expression import *
from wick.convenience import *
from wick.wick import apply_wick


class ATermTest(unittest.TestCase):
    def test_connected(self):
        H1 = one_e("f", ["occ", "vir"], norder=True)
        H2 = two_e("I", ["occ", "vir"], norder=True)

        H = H1 + H2
        bra = braE2("occ", "vir", "occ", "vir")
        C0 = E0("c")
        C1 = E1("c", ["occ"], ["vir"])
        C2 = E2("c", ["occ"], ["vir"])
        ket = C0 + C1 + C2

        HC = H*ket
        S = bra*HC
        out = apply_wick(S)
        out.resolve()
        final = AExpression(Ex=out)
        out = [at.connected() for at in final.terms]
        ref = [True]*19
        ref[0] = ref[1] = ref[2] = ref[3] = False
        self.assertTrue(ref == out)

    def test_reducible(self):
        H1 = one_e("f", ["occ", "vir"], norder=True)
        H2 = two_e("I", ["occ", "vir"], norder=True)

        H = H1 + H2
        bra = braE2("occ", "vir", "occ", "vir")
        C0 = E0("c")
        C1 = E1("c", ["occ"], ["vir"])
        C2 = E2("c", ["occ"], ["vir"])
        ket = C0 + C1 + C2

        HC = H*ket
        S = bra*HC
        out = apply_wick(S)
        out.resolve()
        final = AExpression(Ex=out)
        out = [at.reducible() for at in final.terms]
        ref = [True]*19
        ref[4] = False
        ref[13] = ref[14] = ref[15] = ref[16] = ref[17] = ref[18] = False
        self.assertTrue(ref == out)


if __name__ == '__main__':
    unittest.main()
