import unittest

from wick.index import Idx
from wick.operator import FOperator, BOperator, Projector
from wick.operator import Delta, Tensor, Sigma


class OperatorTest(unittest.TestCase):
    def test_foperator(self):
        i = Idx(0, "occ")
        j = Idx(1, "occ")
        O1 = FOperator(i, False)
        O2 = FOperator(i, False)
        O3 = FOperator(j, False)
        O4 = FOperator(j, True)
        self.assertTrue(O1 == O2)
        self.assertTrue(O1 != O3)
        self.assertTrue(O1 != O4)
        self.assertTrue(O2 != O4)
        self.assertTrue(O1.qp_creation())
        self.assertTrue(O4.qp_annihilation())

    def test_boperator(self):
        I = Idx(0, "nm", fermion=False)
        J = Idx(1, "nm", fermion=False)
        O1 = BOperator(I, False)
        O2 = BOperator(I, False)
        O3 = BOperator(J, False)
        O4 = BOperator(J, True)
        self.assertTrue(O1 == O2)
        self.assertTrue(O1 != O3)
        self.assertTrue(O1 != O4)
        self.assertTrue(O2 != O4)
        self.assertTrue(O4.qp_creation())
        self.assertTrue(O1.qp_annihilation())

    def test_projector(self):
        P1 = Projector()
        P2 = Projector()
        P3 = P1.copy()
        self.assertFalse(P1 != P2)
        self.assertTrue(P1.dagger() == P2)
        self.assertTrue(P1 == P3)

    def test_tensor(self):
        i = Idx(0, "occ")
        a = Idx(0, "vir")
        T0 = Tensor([i], "g")
        T1 = Tensor([i, a], "g")
        T2 = Tensor([i, a], "f")
        T3 = Tensor([i, a], "f")
        T4 = Tensor([a, i], "f")
        self.assertTrue(T2 == T3)
        self.assertTrue(T2 != T4)
        self.assertTrue(T1 != T3)
        self.assertTrue(T1 != T4)

        self.assertTrue(T0 < T1)
        self.assertTrue(T2 <= T3)
        self.assertFalse(T2 < T3)
        self.assertTrue(T2 > T0)
        self.assertTrue(T4 >= T3)

    def test_sigma(self):
        i = Idx(0, "occ")
        j = Idx(1, "occ")
        a = Idx(0, "vir")
        S1 = Sigma(i)
        S2 = Sigma(i)
        S3 = Sigma(j)
        S4 = Sigma(a)
        self.assertTrue(S2 == S1)
        self.assertTrue(S2 != S3)
        self.assertTrue(S2 != S4)
        self.assertTrue(S3 != S4)
        self.assertTrue(S1 < S3)
        self.assertTrue(S1 <= S3)
        self.assertTrue(S1 <= S2)
        self.assertTrue(S1 >= S2)
        self.assertTrue(S4 > S2)

    def test_delta(self):
        i = Idx(0, "occ")
        j = Idx(1, "occ")
        a = Idx(0, "vir")
        b = Idx(1, "vir")
        D1 = Delta(i, j)
        D2 = Delta(i, j)
        D3 = Delta(j, i)
        D4 = Delta(a, b)
        self.assertTrue(D1 == D2)
        self.assertTrue(D1 == D3)
        self.assertTrue(D1 != D4)

    def test_dagger(self):
        i = Idx(0, "occ")
        O1 = FOperator(i, False)
        O2 = FOperator(i, True)
        self.assertTrue(O1.dagger() == O2)
        self.assertTrue(O2.dagger() == O1)

        x = Idx(0, "nm", fermion=False)
        Ob1 = BOperator(x, False)
        Ob2 = BOperator(x, True)
        self.assertTrue(Ob1.dagger() == Ob2)
        self.assertTrue(Ob2.dagger() == Ob1)

    def test_string(self):
        P1 = Projector()
        self.assertTrue(str(P1) == "P")

        i = Idx(0, "occ")
        a = Idx(0, "vir")
        O1 = FOperator(i, False)
        O2 = FOperator(i, True)
        O3 = FOperator(a, False)
        O4 = FOperator(a, True)
        so1 = "a_0(occ)"
        so2 = "a^{\\dagger}_0(occ)"
        so3 = "a_0(vir)"
        so4 = "a^{\\dagger}_0(vir)"
        self.assertTrue(str(O1) == so1)
        self.assertTrue(str(O2) == so2)
        self.assertTrue(str(O3) == so3)
        self.assertTrue(str(O4) == so4)

        x = Idx(0, "nm", fermion=False)
        Ob1 = BOperator(x, False)
        Ob2 = BOperator(x, True)
        sob1 = "b_0(nm)"
        sob2 = "b^{\\dagger}_0(nm)"
        self.assertTrue(str(Ob1) == sob1)
        self.assertTrue(str(Ob2) == sob2)

        T1 = Tensor([i, a], "g")
        st1 = "g_{00}"
        self.assertTrue(str(T1) == st1)

        S1 = Sigma(i)
        ss1 = "\\sum_{0}"
        self.assertTrue(str(S1) == ss1)

        j = Idx(1, "occ")
        D1 = Delta(i, j)
        sd1 = "\\delta_{0,1}"
        self.assertTrue(str(D1) == sd1)

    def test_inc(self):
        i = Idx(0, "occ")
        j = Idx(1, "occ")
        a = Idx(0, "vir")
        I = Idx(0, "nm", fermion=False)
        J = Idx(1, "nm", fermion=False)

        iii = 3

        # Fermion operator
        O1 = FOperator(i, False)._inc(iii)
        self.assertTrue(O1.idx.index == iii)

        # Boson operator
        O2 = BOperator(I, False)._inc(iii)
        O3 = BOperator(J, True)._inc(iii)
        self.assertTrue(O2.idx.index == iii)
        self.assertTrue(O3.idx.index == iii + 1)

        # Projector
        P1 = Projector()
        P2 = P1._inc(iii)
        self.assertTrue(P1 == P2)

        # tensor
        T1 = Tensor([i, a], "g")._inc(iii)
        self.assertTrue(T1.indices[0].index == iii)
        self.assertTrue(T1.indices[1].index == iii)

        # sigma
        S3 = Sigma(j)._inc(iii)
        self.assertTrue(S3.idx.index == iii + 1)

        # delta
        D1 = Delta(i, j)._inc(iii)
        self.assertTrue(D1.i1.index == iii)
        self.assertTrue(D1.i2.index == iii + 1)


if __name__ == '__main__':
    unittest.main()
