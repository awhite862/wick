import unittest
from fractions import Fraction

from wick.index import Idx
from wick.operator import Sigma, Tensor
from wick.expression import Term, Expression, ATerm, AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, get_sym
from wick.convenience import ketE1, ketE2, braE1, braE2


class SCRulesTest(unittest.TestCase):
    def test_0d0(self):
        # 0 differences, constant
        t = Term(1.0, [], [Tensor([], 'I')], [], [])
        e = Expression([t])
        x = apply_wick(e)
        self.assertTrue(x == e)

    def test_0d1(self):
        # 0 differences 1-electron operator
        e = one_e("f", ["occ", "vir"])
        x = apply_wick(e)
        x.resolve()
        out = AExpression(Ex=x)
        i = Idx(0, "occ")
        tr1 = ATerm(
            scalar=1, sums=[Sigma(i)],
            tensors=[Tensor([i, i], "f")])
        ref = AExpression(terms=[tr1])
        self.assertTrue(ref.pmatch(out))

    def test_0d2(self):
        # 0 differences 2-electron operator
        e = two_e("I", ["occ", "vir"])
        e = Expression(e.terms[0:1])
        x = apply_wick(e)
        x.resolve()
        out = AExpression(Ex=x)

        i = Idx(0, "occ")
        j = Idx(1, "occ")
        tr1 = ATerm(
            scalar=Fraction(1, 2), sums=[Sigma(i), Sigma(j)],
            tensors=[Tensor([i, j, i, j], "I", sym=get_sym(True))])
        ref = AExpression(terms=[tr1])
        self.assertTrue(ref.pmatch(out))

    def test_1d0a(self):
        # 1 difference, constant
        t = Term(1.0, [], [Tensor([], 'I')], [], [])
        e = Expression([t])
        ket = ketE1("occ", "vir")
        x = apply_wick(e*ket)
        self.assertTrue(len(x.terms) == 0)

    def test_1d1a(self):
        # 1 differences 1-electron operator
        e = one_e("f", ["occ", "vir"])
        ket = ketE1("occ", "vir")
        x = apply_wick(e*ket)
        x.resolve()
        out = AExpression(Ex=x)

        i = Idx(0, "occ")
        a = Idx(0, "vir")
        tr1 = ATerm(
            scalar=1, sums=[],
            tensors=[Tensor([i, a], "f"), Tensor([i, a], "")])
        ref = AExpression(terms=[tr1])
        self.assertTrue(ref.pmatch(out))

    def test_1d2a(self):
        # 1 difference, 2-electron operator
        e = two_e("I", ["occ", "vir"])
        ket = ketE1("occ", "vir")
        x = apply_wick(e*ket)
        x.resolve()
        out = AExpression(Ex=x)

        i = Idx(0, "occ")
        j = Idx(1, "occ")
        a = Idx(0, "vir")
        tr1 = ATerm(
            scalar=1, sums=[Sigma(j)],
            tensors=[Tensor([i, j, a, j], "I", sym=get_sym(True)),
                     Tensor([i, a], "")])
        ref = AExpression(terms=[tr1])
        self.assertTrue(ref.pmatch(out))

    def test_1d0b(self):
        # 1 difference, constant
        t = Term(1.0, [], [Tensor([], 'I')], [], [])
        e = Expression([t])
        bra = braE1("occ", "vir")
        x = apply_wick(bra*e)
        self.assertTrue(len(x.terms) == 0)

    def test_1d1b(self):
        # 1 differences 1-electron operator
        e = one_e("f", ["occ", "vir"])
        bra = braE1("occ", "vir")
        x = apply_wick(bra*e)
        x.resolve()
        out = AExpression(Ex=x)

        i = Idx(0, "occ")
        a = Idx(0, "vir")
        tr1 = ATerm(
            scalar=1, sums=[],
            tensors=[Tensor([a, i], ""), Tensor([a, i], "f")])
        ref = AExpression(terms=[tr1])
        self.assertTrue(ref.pmatch(out))

    def test_1d2b(self):
        # 1 difference, 2-electron operator (compressed)
        e = two_e("I", ["occ", "vir"], compress=True)
        bra = braE1("occ", "vir")
        x = apply_wick(bra*e)
        x.resolve()
        out = AExpression(Ex=x)

        i = Idx(0, "occ")
        j = Idx(1, "occ")
        a = Idx(0, "vir")
        tr1 = ATerm(
            scalar=1, sums=[Sigma(j)],
            tensors=[Tensor([a, i], ""),
                     Tensor([a, j, i, j], "I", sym=get_sym(True))])
        ref = AExpression(terms=[tr1])
        self.assertTrue(ref.pmatch(out))

    def test_2d1a(self):
        # 2 differences, 1-electron operator
        e = one_e("f", ["occ", "vir"])
        ket = ketE2("occ", "vir", "occ", "vir")
        x = apply_wick(e*ket)
        self.assertTrue(len(x.terms) == 0)

    def test_2d2a(self):
        # 2 differences, 2-electron operator
        e = two_e("I", ["occ", "vir"])
        ket = ketE2("occ", "vir", "occ", "vir")
        x = apply_wick(e*ket)
        x.resolve()
        out = AExpression(Ex=x)

        i = Idx(0, "occ")
        j = Idx(1, "occ")
        a = Idx(0, "vir")
        b = Idx(1, "vir")
        tr1 = ATerm(
            scalar=1, sums=[],
            tensors=[Tensor([i, j, a, b], "I", sym=get_sym(True)),
                     Tensor([i, j, a, b], "", sym=get_sym(True))])
        ref = AExpression(terms=[tr1])
        self.assertTrue(ref.pmatch(out))

    def test_2d1b(self):
        # 2 differences, 1-electron operator
        e = one_e("f", ["occ", "vir"])
        bra = braE2("occ", "vir", "occ", "vir")
        x = apply_wick(bra*e)
        self.assertTrue(len(x.terms) == 0)

    def test_2d2b(self):
        # 2 differences, 2-electron operator (compressed)
        e = two_e("I", ["occ", "vir"], compress=True, norder=True)
        bra = braE2("occ", "vir", "occ", "vir")
        x = apply_wick(bra*e)
        x.resolve()
        out = AExpression(Ex=x)

        i = Idx(0, "occ")
        j = Idx(1, "occ")
        a = Idx(0, "vir")
        b = Idx(1, "vir")
        tr1 = ATerm(
            scalar=1, sums=[],
            tensors=[Tensor([a, b, i, j], "", sym=get_sym(True)),
                     Tensor([a, b, i, j], "I", sym=get_sym(True))])
        ref = AExpression(terms=[tr1])
        self.assertTrue(ref.pmatch(out))

    def test_2d1c(self):
        # 2 differences, 1-electron operator
        e = one_e("f", ["occ", "vir"], norder=True)
        bra = braE1("occ", "vir")
        ket = ketE1("occ", "vir")
        x = apply_wick(bra*e*ket)
        x.resolve()
        out = AExpression(Ex=x)
        out = out.get_connected()  # terms with truly 2 differences
        self.assertTrue(len(out.terms) == 0)

    def test_2d2c(self):
        # 2 differences, 2-electron operator
        e = two_e("I", ["occ", "vir"], norder=True)
        bra = braE1("occ", "vir")
        ket = ketE1("occ", "vir")
        x = apply_wick(bra*e*ket)
        x.resolve()
        out = AExpression(Ex=x)

        i = Idx(0, "occ")
        j = Idx(1, "occ")
        a = Idx(0, "vir")
        b = Idx(1, "vir")
        tr1 = ATerm(
            scalar=1, sums=[],
            tensors=[Tensor([a, i], ""),
                     Tensor([a, j, i, b], "I", sym=get_sym(True)),
                     Tensor([j, b], "")])
        ref = AExpression(terms=[tr1])
        self.assertTrue(ref.pmatch(out))


if __name__ == '__main__':
    unittest.main()
