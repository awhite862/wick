import unittest
from wick.index import Idx
from wick.operator import Delta, Tensor, TensorSym
from wick.expression import Term, ATerm, Expression, AExpression
from wick.wick import apply_wick
from wick.convenience import get_sym, get_sym_ip2, get_sym_ea2
from wick.convenience import E1, E2, Eip1, Eea1, Eip2, Eea2, P1, P2
from wick.convenience import EPS1, EP1ip1, EP1ea1, EPS2
from wick.convenience import braE1, braE2
from wick.convenience import braEip1, braEip2, braEdip1
from wick.convenience import braEea1, braEea2, braEdea1
from wick.convenience import ketE1, ketE2
from wick.convenience import ketEip1, ketEip2, ketEdip1
from wick.convenience import ketEea1, ketEea2, ketEdea1
from wick.convenience import braP1, braP2, braP1E1, braP1Eip1, braP1Eea1
from wick.convenience import ketP1, ketP2, ketP1E1, ketP1Eip1, ketP1Eea1
from wick.convenience import braP2E1, one_p, two_p, ep11


class ConvenienceTest(unittest.TestCase):
    def testE1(self):
        bra = braE1("occ", "vir")
        ket = ketE1("occ", "vir")
        out = apply_wick(bra*ket)
        aout = AExpression(Ex=out)

        i = Idx(0, "occ")
        a = Idx(0, "vir")
        j = Idx(1, "occ")
        b = Idx(1, "vir")
        tr1 = Term(
            1, [], [Tensor([a, i], ""), Tensor([j, b], "")],
            [], [Delta(i, j), Delta(a, b)])
        ref = Expression([tr1])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

        op = E1("A", ["occ"], ["vir"])
        out = apply_wick(bra*op)
        out.resolve()
        aout = AExpression(Ex=out)
        ext = Tensor([a, i], "")
        ten = Tensor([a, i], "A")
        at1 = ATerm(scalar=1, sums=[], tensors=[ext, ten])
        self.assertTrue(len(aout.terms) == 1)
        self.assertTrue(at1 == aout.terms[0])

    def testE2(self):
        bra = braE2("occ", "vir", "occ", "vir")
        ket = ketE2("occ", "vir", "occ", "vir")
        out = apply_wick(bra*ket)
        aout = AExpression(Ex=out)

        i = Idx(0, "occ")
        a = Idx(0, "vir")
        j = Idx(1, "occ")
        b = Idx(1, "vir")
        k = Idx(2, "occ")
        c = Idx(2, "vir")
        l = Idx(3, "occ")
        d = Idx(3, "vir")
        tensors = [
            Tensor([a, b, i, j], ""),
            Tensor([k, l, c, d], "")]
        tr1 = Term(
            1, [], tensors, [],
            [Delta(i, k), Delta(j, l), Delta(a, c), Delta(b, d)])
        tr2 = Term(
            -1, [], tensors, [],
            [Delta(i, l), Delta(j, k), Delta(a, c), Delta(b, d)])
        tr3 = Term(
            -1, [], tensors, [],
            [Delta(i, k), Delta(j, l), Delta(a, d), Delta(b, c)])
        tr4 = Term(
            1, [], tensors, [],
            [Delta(i, l), Delta(j, k), Delta(a, d), Delta(b, c)])
        ref = Expression([tr1, tr2, tr3, tr4])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

        op = E2("A", ["occ"], ["vir"])
        out = apply_wick(bra*op)
        out.resolve()
        aout = AExpression(Ex=out)
        ext = Tensor([a, b, i, j], "")
        ten = Tensor([a, b, i, j], "A", sym=get_sym(True))
        at1 = ATerm(scalar=1, sums=[], tensors=[ext, ten])
        self.assertTrue(len(aout.terms) == 1)
        self.assertTrue(at1.pmatch(aout.terms[0]))

    def testEip1(self):
        bra = braEip1("occ")
        ket = ketEip1("occ")
        out = apply_wick(bra*ket)
        aout = AExpression(Ex=out)

        i = Idx(0, "occ")
        j = Idx(1, "occ")
        tr1 = Term(
            1, [], [Tensor([i], ""), Tensor([j], "")],
            [], [Delta(i, j)])
        ref = Expression([tr1])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

        op = Eip1("A", ["occ"])
        out = apply_wick(bra*op)
        out.resolve()
        aout = AExpression(Ex=out)
        ext = Tensor([i], "")
        ten = Tensor([i], "A")
        at1 = ATerm(scalar=1, sums=[], tensors=[ext, ten])
        self.assertTrue(len(aout.terms) == 1)
        self.assertTrue(at1 == aout.terms[0])

    def testEip2(self):
        bra = braEip2("occ", "occ", "vir")
        ket = ketEip2("occ", "occ", "vir")
        out = apply_wick(bra*ket)
        aout = AExpression(Ex=out)

        i = Idx(0, "occ")
        a = Idx(0, "vir")
        j = Idx(1, "occ")
        k = Idx(2, "occ")
        c = Idx(1, "vir")
        l = Idx(3, "occ")
        tensors = [
            Tensor([a, i, j], ""),
            Tensor([k, l, c], "")]
        tr1 = Term(
            1, [], tensors, [],
            [Delta(i, k), Delta(j, l), Delta(a, c)])
        tr2 = Term(
            -1, [], tensors, [],
            [Delta(i, l), Delta(j, k), Delta(a, c)])
        ref = Expression([tr1, tr2])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

        op = Eip2("A", ["occ"], ["vir"])
        out = apply_wick(bra*op)
        out.resolve()
        aout = AExpression(Ex=out)
        ext = Tensor([a, i, j], "")
        ten = Tensor([a, i, j], "A", sym=get_sym_ip2())
        at1 = ATerm(scalar=1, sums=[], tensors=[ext, ten])
        self.assertTrue(len(aout.terms) == 1)
        self.assertTrue(at1.pmatch(aout.terms[0]))

    def testEdip1(self):
        bra = braEdip1("occ", "occ")
        ket = ketEdip1("occ", "occ")
        out = apply_wick(bra*ket)
        aout = AExpression(Ex=out)

        i = Idx(0, "occ")
        j = Idx(1, "occ")
        k = Idx(2, "occ")
        l = Idx(3, "occ")
        tensors = [
            Tensor([i, j], ""),
            Tensor([k, l], "")]
        tr1 = Term(
            1, [], tensors, [],
            [Delta(i, k), Delta(j, l)])
        tr2 = Term(
            -1, [], tensors, [],
            [Delta(i, l), Delta(j, k)])
        ref = Expression([tr1, tr2])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

    def testEea1(self):
        bra = braEea1("vir")
        ket = ketEea1("vir")
        out = apply_wick(bra*ket)
        aout = AExpression(Ex=out)

        a = Idx(0, "vir")
        b = Idx(1, "vir")
        tr1 = Term(
            1, [], [Tensor([a], ""), Tensor([b], "")],
            [], [Delta(a, b)])
        ref = Expression([tr1])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

        op = Eea1("A", ["vir"])
        out = apply_wick(bra*op)
        out.resolve()
        aout = AExpression(Ex=out)
        ext = Tensor([a], "")
        ten = Tensor([a], "A")
        at1 = ATerm(scalar=1, sums=[], tensors=[ext, ten])
        self.assertTrue(at1 == aout.terms[0])

    def testEea2(self):
        bra = braEea2("occ", "vir", "vir")
        ket = ketEea2("occ", "vir", "vir")
        out = apply_wick(bra*ket)
        aout = AExpression(Ex=out)

        i = Idx(0, "occ")
        a = Idx(0, "vir")
        b = Idx(1, "vir")
        k = Idx(2, "occ")
        c = Idx(2, "vir")
        d = Idx(3, "vir")
        tensors = [
            Tensor([i, a, b], ""),
            Tensor([c, d, k], "")]
        tr1 = Term(
            1, [], tensors, [],
            [Delta(i, k), Delta(a, c), Delta(b, d)])
        tr2 = Term(
            -1, [], tensors, [],
            [Delta(i, k), Delta(a, d), Delta(b, c)])
        ref = Expression([tr1, tr2])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

        op = Eea2("A", ["occ"], ["vir"])
        out = apply_wick(bra*op)
        out.resolve()
        aout = AExpression(Ex=out)
        ext = Tensor([a, b, i], "")
        ten = Tensor([a, b, i], "A", sym=get_sym_ea2())
        at1 = ATerm(scalar=1, sums=[], tensors=[ext, ten])
        self.assertTrue(len(aout.terms) == 1)
        self.assertTrue(at1.pmatch(aout.terms[0]))

    def testEdea1(self):
        bra = braEdea1("vir", "vir")
        ket = ketEdea1("vir", "vir")
        out = apply_wick(bra*ket)
        aout = AExpression(Ex=out)

        a = Idx(0, "vir")
        b = Idx(1, "vir")
        c = Idx(2, "vir")
        d = Idx(3, "vir")
        tensors = [
            Tensor([a, b], ""),
            Tensor([c, d], "")]
        tr1 = Term(
            1, [], tensors, [],
            [Delta(a, c), Delta(b, d)])
        tr2 = Term(
            -1, [], tensors, [],
            [Delta(a, d), Delta(b, c)])
        ref = Expression([tr1, tr2])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

    def testP1(self):
        bra = braP1("nm")
        ket = ketP1("nm")
        out = apply_wick(bra*ket)
        aout = AExpression(Ex=out)

        x = Idx(0, "nm", fermion=False)
        y = Idx(1, "nm", fermion=False)
        tensors = [
            Tensor([x], ""),
            Tensor([y], "")]
        tr1 = Term(
            1, [], tensors, [], [Delta(x, y)])
        ref = Expression([tr1])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

        op = P1("A", ["nm"])
        out = apply_wick(bra*op)
        out.resolve()
        aout = AExpression(Ex=out)
        ext = Tensor([x], "")
        ten = Tensor([x], "A")
        at1 = ATerm(scalar=1, sums=[], tensors=[ext, ten])
        self.assertTrue(at1 == aout.terms[0])

    def testP2(self):
        bra = braP2("nm")
        ket = ketP2("nm")
        out = apply_wick(bra*ket)
        aout = AExpression(Ex=out)

        x = Idx(0, "nm", fermion=False)
        y = Idx(1, "nm", fermion=False)
        u = Idx(2, "nm", fermion=False)
        v = Idx(3, "nm", fermion=False)
        tensors = [
            Tensor([x, y], ""),
            Tensor([u, v], "")]
        tr1 = Term(
            1, [], tensors, [], [Delta(x, u), Delta(y, v)])
        tr2 = Term(
            1, [], tensors, [], [Delta(x, v), Delta(y, u)])
        ref = Expression([tr1, tr2])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

        op = P2("A", ["nm"])
        out = apply_wick(bra*op)
        out.resolve()
        aout = AExpression(Ex=out)
        sym = TensorSym([(0, 1), (1, 0)], [1, 1])
        ext = Tensor([x, y], "")
        t1 = Tensor([x, y], "A", sym=sym)
        at1 = ATerm(scalar=1, sums=[], tensors=[ext, t1])
        self.assertTrue(at1 == aout.terms[0])

    def testP1E1(self):
        bra = braP1E1("nm", "occ", "vir")
        ket = ketP1E1("nm", "occ", "vir")
        out = apply_wick(bra*ket)
        aout = AExpression(Ex=out)

        x = Idx(0, "nm", fermion=False)
        y = Idx(1, "nm", fermion=False)
        i = Idx(0, "occ")
        a = Idx(0, "vir")
        j = Idx(1, "occ")
        b = Idx(1, "vir")
        tensors = [
            Tensor([x, i, a], ""),
            Tensor([y, b, j], "")]
        tr1 = Term(
            1, [], tensors, [], [Delta(x, y), Delta(i, j), Delta(a, b)])
        ref = Expression([tr1])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

        op = EPS1("A", ["nm"], ["occ"], ["vir"])
        out = apply_wick(bra*op)
        out.resolve()
        aout = AExpression(Ex=out)
        ext = Tensor([x, a, i], "")
        t1 = Tensor([x, a, i], "A")
        at1 = ATerm(scalar=1, sums=[], tensors=[ext, t1])
        self.assertTrue(at1 == aout.terms[0])

    def testP1Eip1(self):
        bra = braP1Eip1("nm", "occ")
        ket = ketP1Eip1("nm", "occ")
        out = apply_wick(bra*ket)
        aout = AExpression(Ex=out)

        x = Idx(0, "nm", fermion=False)
        y = Idx(1, "nm", fermion=False)
        i = Idx(0, "occ")
        j = Idx(1, "occ")
        tensors = [
            Tensor([x, i], ""),
            Tensor([y, j], "")]
        tr1 = Term(
            1, [], tensors, [], [Delta(x, y), Delta(i, j)])
        ref = Expression([tr1])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

        op = EP1ip1("A", ["nm"], ["occ"])
        out = apply_wick(bra*op)
        out.resolve()
        aout = AExpression(Ex=out)
        ext = Tensor([x, i], "")
        t1 = Tensor([x, i], "A")
        at1 = ATerm(scalar=1, sums=[], tensors=[ext, t1])
        self.assertTrue(at1 == aout.terms[0])

    def testP1Eea1(self):
        bra = braP1Eea1("nm", "vir")
        ket = ketP1Eea1("nm", "vir")
        out = apply_wick(bra*ket)
        aout = AExpression(Ex=out)

        x = Idx(0, "nm", fermion=False)
        y = Idx(1, "nm", fermion=False)
        a = Idx(0, "vir")
        b = Idx(1, "vir")
        tensors = [
            Tensor([x, a], ""),
            Tensor([y, b], "")]
        tr1 = Term(
            1, [], tensors, [], [Delta(x, y), Delta(a, b)])
        ref = Expression([tr1])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

        op = EP1ea1("A", ["nm"], ["vir"])
        out = apply_wick(bra*op)
        out.resolve()
        aout = AExpression(Ex=out)
        ext = Tensor([x, a], "")
        t1 = Tensor([x, a], "A")
        at1 = ATerm(scalar=1, sums=[], tensors=[ext, t1])
        self.assertTrue(at1 == aout.terms[0])

    def testP2E1(self):
        bra = braP2E1("nm", "nm", "occ", "vir")
        op = EPS2("A", ["nm"], ["occ"], ["vir"])
        out = apply_wick(bra*op)
        out.resolve()

        x = Idx(0, "nm", fermion=False)
        y = Idx(1, "nm", fermion=False)
        a = Idx(0, "vir")
        i = Idx(0, "occ")
        sym = TensorSym([(0, 1, 2, 3), (1, 0, 2, 3)], [1, 1])
        aout = AExpression(Ex=out)
        ext = Tensor([x, y, a, i], "")
        t1 = Tensor([x, y, a, i], "A", sym=sym)
        at1 = ATerm(scalar=1, sums=[], tensors=[ext, t1])
        self.assertTrue(at1 == aout.terms[0])

    def testP1op(self):
        op = one_p("Hp", name2="Hq")
        bra = braP1("nm")
        ex = apply_wick(bra*op)
        ex.resolve()
        out = AExpression(Ex=ex)

        x = Idx(0, "nm", fermion=False)
        tr1 = ATerm(
            scalar=1, sums=[],
            tensors=[Tensor([x], "Hq"), Tensor([x], "")])
        ref = AExpression(terms=[tr1])
        self.assertTrue(ref.pmatch(out))

        ket = ketP1("nm")
        ex = apply_wick(op*ket)
        ex.resolve()
        out = AExpression(Ex=ex)
        tr1 = ATerm(
            scalar=1, sums=[],
            tensors=[Tensor([x], "Hp"), Tensor([x], "")])
        ref = AExpression(terms=[tr1])
        self.assertTrue(ref.pmatch(out))

    def testP2op(self):
        op = two_p("H")
        bra = braP1("nm")
        ket = ketP1("nm")
        ex = apply_wick(bra*op*ket)
        ex.resolve()
        out = AExpression(Ex=ex)

        x = Idx(0, "nm", fermion=False)
        y = Idx(1, "nm", fermion=False)
        sym = TensorSym([(0, 1), (1, 0)], [1, 1])
        tensors = [
            Tensor([x], ""), Tensor([x, y], "H", sym=sym), Tensor([y], "")]
        tr1 = ATerm(scalar=1, sums=[], tensors=tensors)
        ref = AExpression(terms=[tr1])
        self.assertTrue(ref.pmatch(out))

    def testEP11op(self):
        op = ep11("Hp", ["occ", "vir"], ["nm"], name2="Hq")
        bra = braP1E1("nm", "occ", "vir")
        ex = apply_wick(bra*op)
        ex.resolve()
        out = AExpression(Ex=ex)

        x = Idx(0, "nm", fermion=False)
        a = Idx(0, "vir")
        i = Idx(0, "occ")
        tr1 = ATerm(
            scalar=1, sums=[],
            tensors=[Tensor([x, a, i], ""), Tensor([x, a, i], "Hq")])
        ref = AExpression(terms=[tr1])
        self.assertTrue(ref.pmatch(out))

        ket = ketP1E1("nm", "occ", "vir")
        ex = apply_wick(op*ket)
        ex.resolve()
        out = AExpression(Ex=ex)

        tr1 = ATerm(
            scalar=1, sums=[],
            tensors=[Tensor([x, i, a], "Hp"), Tensor([x, i, a], "")])
        ref = AExpression(terms=[tr1])
        self.assertTrue(ref.pmatch(out))


if __name__ == '__main__':
    unittest.main()
