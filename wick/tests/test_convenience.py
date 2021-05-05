import unittest
from wick.index import Idx
from wick.operator import Delta, Tensor
from wick.expression import Term, Expression, AExpression
from wick.wick import apply_wick
from wick.convenience import braE1, braE2
from wick.convenience import braEip1, braEip2, braEdip1
from wick.convenience import braEea1, braEea2, braEdea1
from wick.convenience import ketE1, ketE2
from wick.convenience import ketEip1, ketEip2, ketEdip1
from wick.convenience import ketEea1, ketEea2, ketEdea1
from wick.convenience import braP1, braP2, braP1E1, braP1Eip1, braP1Eea1
from wick.convenience import ketP1, ketP2, ketP1E1, ketP1Eip1, ketP1Eea1


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
            1, [], [Tensor([i, a], ""), Tensor([b, j], "")],
            [], [Delta(i, j), Delta(a, b)])
        ref = Expression([tr1])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

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
            Tensor([i, j, a, b], ""),
            Tensor([c, d, k, l], "")]
        tr1 = Term(
            1, [],  tensors, [],
            [Delta(i, k), Delta(j, l), Delta(a, c), Delta(b, d)])
        tr2 = Term(
            -1, [],  tensors, [],
            [Delta(i, l), Delta(j, k), Delta(a, c), Delta(b, d)])
        tr3 = Term(
            -1, [],  tensors, [],
            [Delta(i, k), Delta(j, l), Delta(a, d), Delta(b, c)])
        tr4 = Term(
            1, [],  tensors, [],
            [Delta(i, l), Delta(j, k), Delta(a, d), Delta(b, c)])
        ref = Expression([tr1, tr2, tr3, tr4])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

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
            Tensor([i, j, a], ""),
            Tensor([c, k, l], "")]
        tr1 = Term(
            1, [],  tensors, [],
            [Delta(i, k), Delta(j, l), Delta(a, c)])
        tr2 = Term(
            -1, [],  tensors, [],
            [Delta(i, l), Delta(j, k), Delta(a, c)])
        ref = Expression([tr1, tr2])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

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
            1, [],  tensors, [],
            [Delta(i, k), Delta(j, l)])
        tr2 = Term(
            -1, [],  tensors, [],
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
            1, [],  tensors, [],
            [Delta(i, k), Delta(a, c), Delta(b, d)])
        tr2 = Term(
            -1, [],  tensors, [],
            [Delta(i, k), Delta(a, d), Delta(b, c)])
        ref = Expression([tr1, tr2])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

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
            1, [],  tensors, [],
            [Delta(a, c), Delta(b, d)])
        tr2 = Term(
            -1, [],  tensors, [],
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
            1, [],  tensors, [], [Delta(x, y)])
        ref = Expression([tr1])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

    def testP2(self):
        bra = braP2("nm", "nm")
        ket = ketP2("nm", "nm")
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
            1, [],  tensors, [], [Delta(x, u), Delta(y,v)])
        tr2 = Term(
            1, [],  tensors, [], [Delta(x, v), Delta(y,u)])
        ref = Expression([tr1, tr2])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

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
            1, [],  tensors, [], [Delta(x, y), Delta(i, j), Delta(a, b)])
        ref = Expression([tr1])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

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
            1, [],  tensors, [], [Delta(x, y), Delta(i, j)])
        ref = Expression([tr1])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))

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
            1, [],  tensors, [], [Delta(x, y), Delta(a, b)])
        ref = Expression([tr1])
        aref = AExpression(Ex=ref)
        self.assertTrue(aout.pmatch(aref))


if __name__ == '__main__':
    unittest.main()
