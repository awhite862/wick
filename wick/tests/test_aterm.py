import unittest

from wick.index import Idx
from wick.operator import Tensor, Delta, Sigma, tensor_from_delta
from wick.expression import AExpression, ATerm
from wick.convenience import one_e, two_e, braE2, braE1, ketE1
from wick.convenience import E0, E1, E2
from wick.wick import apply_wick


class ATermTest(unittest.TestCase):
    def test_merge_external(self):
        bra = braE1("occ", "vir")
        ket = ketE1("occ", "vir")
        out = apply_wick(bra*ket)
        aout = AExpression(Ex=out)
        aterm = aout.terms[0]
        aterm.merge_external()

        i = Idx(0, "occ")
        a = Idx(0, "vir")
        j = Idx(1, "occ")
        b = Idx(1, "vir")
        tensors = [
            Tensor([a, i, j, b], ""),
            tensor_from_delta(Delta(i, j)),
            tensor_from_delta(Delta(a, b))]
        aref = ATerm(scalar=1, sums=[], tensors=tensors)
        self.assertTrue(aterm == aref)

    def test_tensor_sort(self):
        i = Idx(0, "occ")
        j = Idx(1, "occ")
        a = Idx(0, "vir")
        tensors = [
            Tensor([j, i], 'f'), Tensor([a, i], ''), Tensor([a, j], "t")]
        st = [tensors[1], tensors[0], tensors[2]]
        sigmas = [Sigma(j)]
        tt = ATerm(scalar=1.0, sums=sigmas, tensors=tensors)
        tt.sort_tensors()
        for ref, out in zip(st, tt.tensors):
            self.assertTrue(ref == out)

    def test_term_map(self):
        s = 1.0
        i = Idx(0, "occ")
        j = Idx(1, "occ")
        sums = [Sigma(i), Sigma(j)]
        tensors = [Tensor([i, j], 'f')]
        t1 = ATerm(s, sums, tensors)
        sums = [Sigma(i), Sigma(j)]
        tensors = [Tensor([j, i], 'f')]
        t2 = ATerm(s, sums, tensors)
        sums = [Sigma(j), Sigma(i)]
        tensors = [Tensor([i, j], 'f')]
        t3 = ATerm(s, sums, tensors)

        self.assertTrue(t1.match(t2))
        self.assertTrue(t1.match(t3))
        self.assertTrue(t2.match(t3))

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

    def test_eq(self):
        s = 1.0
        i = Idx(0, "occ")
        j = Idx(1, "occ")
        sums = [Sigma(i), Sigma(j)]
        tensors = [Tensor([i, j], 'f')]
        t1 = ATerm(s, sums, tensors)
        t0 = t1.copy()
        sums = [Sigma(i), Sigma(j)]
        tensors = [Tensor([j, i], 'f'), Tensor([i, j], 'g')]
        t2 = ATerm(s, sums, tensors)
        sums = [Sigma(i)]
        tensors = [Tensor([i, j], 'f')]
        t3 = ATerm(s, sums, tensors)
        sums = [Sigma(i)]
        tensors = [Tensor([i, j], 'g')]
        t4 = ATerm(s, sums, tensors)
        self.assertTrue(t1 < t2)
        self.assertFalse(t0 < t1)
        self.assertTrue(t0 <= t1)
        self.assertTrue(t0 >= t1)
        self.assertTrue(t1 > t3)
        self.assertTrue(t3 < t4)


if __name__ == '__main__':
    unittest.main()
