# Copyright (c) 2020-2021 Alec White
# Licensed under the MIT License (see LICENSE for details)
from copy import copy
from itertools import product
from numbers import Number
from .operator import Sigma, Tensor, permute, tensor_from_delta


class TermMap(object):
    """Map indicating the contraction pattern of a given tensor expression

    Attributes:
        data (set): set of tuples indicating a contraction pattern of a tensor expression
    """
    def __init__(self, sums, tensors, occ=None):
        self.data = set()
        sindices = [s.idx for s in sums]
        for ti in tensors:
            ilist = ti.ilist()
            strs = {idx.space: frozenset() for idx in ilist}
            for i, iidx in enumerate(ti.indices):
                space = iidx.space
                istr = str(i)
                summed = iidx in sindices
                for tj in tensors:
                    tjname = tj.name if tj.name else "!"

                    def make_str(x, ss):
                        sout = istr + tjname + x[0]
                        return sout + 'x' if ss else sout

                    jts = [(str(j), jidx) for j, jidx in enumerate(tj.indices) if jidx == iidx]
                    if jts:
                        strs[space] = strs[space].union(
                            frozenset([make_str(x, summed) for x in jts]))
            tiname = ti.name if ti.name else "!"
            #lll = [(k, v) for k, v in sorted(strs.items())]
            lll = sorted(strs.items())
            self.data.add((tiname, tuple(lll)))

    def __eq__(self, other):
        if isinstance(other, TermMap):
            return self.data == other.data
        else:
            return NotImplemented


default_index_key = {"occ": "ijklmnop", "vir": "abcdefgh", "nm": "IJKLMNOP"}


def _resolve(sums, tensors, operators, deltas):
    newdel = [d.copy() for d in deltas]
    newsums = [s.copy() for s in sums]
    newtens = [t.copy() for t in tensors]
    newops = [o.copy() for o in operators]

    # get unique deltas
    newdel = list(set(newdel))

    # Cases:
    #   0 sums over neither index
    #   1 sums over 1st index
    #   2 sums over 2nd index
    #   3 sums over both indices
    def get_case(dd):
        i2 = dd.i2
        i1 = dd.i1
        assert i1.space == i2.space

        #islist = set([s.idx for s in sums])
        islist = {s.idx for s in sums}
        is1 = i1 in islist
        is2 = i2 in islist
        case = 0
        if is1:
            case = 1
        if is2:
            case = (2 if case == 0 else 3)
        return case

    cases = [get_case(dd) for dd in newdel]

    rs = []
    # loop over deltas for case 1 and 2
    for dd, case in zip(newdel, cases):
        i2 = dd.i2
        i1 = dd.i1

        if case == 3:
            continue
        if case == 1:
            dindx = newsums.index(Sigma(i1))
            del newsums[dindx]
        elif case == 2:
            dindx = newsums.index(Sigma(i2))
            del newsums[dindx]
        else:
            assert case == 0

        for i, (ddd, ccc) in enumerate(zip(newdel, cases)):
            if case == 1 and ddd.i1 == i1:
                newdel[i].i1 = i2
                if ccc == 3:
                    cases[i] = 2
                elif ccc == 1:
                    cases[i] = 0
                else:
                    assert False
            elif case == 1 and ddd.i2 == i1:
                newdel[i].i2 = i2
                if ccc == 3:
                    cases[i] = 1
                elif ccc == 2:
                    cases[i] = 0
                else:
                    assert False
            elif case == 2 and ddd.i2 == i2:
                newdel[i].i2 = i1
                if ccc == 3:
                    cases[i] = 1
                elif ccc == 2:
                    cases[i] = 0
                else:
                    assert False
            elif case == 2 and ddd.i1 == i2:
                newdel[i].i1 = i1
                if ccc == 3:
                    cases[i] = 2
                elif ccc == 1:
                    cases[i] = 0
                else:
                    assert False

        for tt in newtens:
            for k, _ in enumerate(tt.indices):
                if case == 1:
                    if tt.indices[k] == i1:
                        tt.indices[k] = i2
                elif case == 2:
                    if tt.indices[k] == i2:
                        tt.indices[k] = i1

        for oo in newops:
            if case == 1:
                if oo.idx == i1:
                    oo.idx = i2
            elif case == 2:
                if oo.idx == i2:
                    oo.idx = i1

        if not (case == 0 and i1 != i2):
            rs.append(dd)

    for d in rs:
        dindx = newdel.index(d)
        del newdel[dindx]
        del cases[dindx]

    # recur if deltas of type 1 or 2 remain
    if 1 in cases or 2 in cases:
        return _resolve(newsums, newtens, newops, newdel)

    # loop over case 3 deltas
    rs = []
    for dd, case in zip(newdel, cases):
        i2 = dd.i2
        i1 = dd.i1

        if case == 3:
            dindx = newsums.index(Sigma(i2))
            del newsums[dindx]
        elif case < 3:
            assert case == 0
        else:
            assert False

        if case == 0:
            continue

        for tt in newtens:
            for k, _ in enumerate(tt.indices):
                if tt.indices[k] == i2:
                    tt.indices[k] = i1

        for oo in newops:
            if oo.idx == i2:
                oo.idx = i1

        rs.append(dd)

    for d in rs:
        dindx = newdel.index(d)
        del newdel[dindx]
    return newsums, newtens, newops, newdel


class Term(object):
    """Term of operators

    Attributes:
        scalar (Number): scalar multiplying the term
        sums (list): list of sums over indices
        tensors (list): list of tensors
        operators (list): list of creation/anihillation operators
        deltas (list): list of delta functions
    """
    def __init__(self, scalar, sums, tensors, operators, deltas, index_key=None):
        self.scalar = scalar
        self.sums = sums
        self.tensors = tensors
        self.operators = operators
        self.deltas = deltas
        self.index_key = index_key

    def resolve(self):
        newsums, newtens, newops, dnew = _resolve(
            self.sums, self.tensors, self.operators, self.deltas)
        self.sums = newsums
        self.tensors = newtens
        self.operators = newops
        self.deltas = dnew

    def __repr__(self):
        out = str(self.scalar)
        for ss in self.sums:
            out += str(ss)
        for dd in self.deltas:
            out += str(dd)
        for tt in self.tensors:
            out += str(tt)
        for oo in self.operators:
            out += str(oo)
        return out

    def __mul__(self, other):
        if isinstance(other, Number):
            new = self.copy()
            new.scalar *= other
            return new
        elif isinstance(other, Term):
            sil1 = set(self.ilist())
            sil2 = set(other.ilist())
            if sil1.intersection(sil2):
                m = max([i.index for i in sil1])
                new = other._inc(m + 1)
            else:
                new = other
            scalar = self.scalar*new.scalar
            sums = self.sums + new.sums
            tensors = self.tensors + new.tensors
            operators = self.operators + new.operators
            deltas = self.deltas + new.deltas
            index_key = other.index_key if self.index_key is None else self.index_key
            return Term(
                scalar, sums, tensors, operators, deltas, index_key=index_key)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            new = self.copy()
            new.scalar *= other
            return new
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Term):
            return self.scalar == other.scalar \
                and set(self.sums) == set(other.sums) \
                and set(self.tensors) == set(other.tensors) \
                and self.operators == other.operators \
                and set(self.deltas) == set(other.deltas)
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def _inc(self, i):
        sums = [s._inc(i) for s in self.sums]
        tensors = [t._inc(i) for t in self.tensors]
        operators = [o._inc(i) for o in self.operators]
        deltas = [d._inc(i) for d in self.deltas]
        return Term(
            self.scalar, sums, tensors, operators,
            deltas, index_key=self.index_key)

    def _idx_map(self):
        if self.index_key is None:
            index_key = default_index_key
        else:
            index_key = self.index_key
        ilist = self.ilist()
        off = {}
        imap = {}
        for idx in ilist:
            s = idx.space
            if s in off:
                o = off[s]
                off[s] += 1
            else:
                o = 0
                off[s] = 1
            imap[idx] = index_key[s][o]
        return imap

    def _print_str(self, with_scalar=True):
        imap = self._idx_map()
        out = str(self.scalar) if with_scalar else str()
        for ss in self.sums:
            out += ss._print_str(imap)
        for dd in self.deltas:
            out += dd._print_str(imap)
        for tt in self.tensors:
            out += tt._print_str(imap)
        for oo in self.operators:
            out += oo._print_str(imap)
        return out

    def ilist(self):
        ilist = set()
        for oo in self.operators:
            if oo.idx is not None:
                ilist.add(oo.idx)
        for tt in self.tensors:
            itlst = set(tt.ilist())
            ilist |= itlst
        for ss in self.sums:
            ilist.add(ss.idx)
        for dd in self.deltas:
            ilist.add(dd.i1)
            ilist.add(dd.i2)
        ret = list(ilist)
        ret.sort()
        return ret

    def copy(self):
        newscalar = copy(self.scalar)
        newsums = [s.copy() for s in self.sums]
        newtensors = [t.copy() for t in self.tensors]
        newoperators = [o.copy() for o in self.operators]
        newdeltas = [d.copy() for d in self.deltas]
        return Term(
            newscalar, newsums, newtensors, newoperators,
            newdeltas, index_key=self.index_key)


class ATerm(object):
    """Abstract term

    Attributes:
        scalar (Number): scalar constant multiplying the term
        sums (list): list of Sums in the term
        tensors (list): list of Tensors
    """
    def __init__(self, scalar=None, sums=None,
                 tensors=None, index_key=None, term=None):
        if term is not None:
            assert len(term.operators) == 0
            if scalar is not None:
                raise Exception("ATerm improperly initialized")
            if sums is not None:
                raise Exception("ATerm improperly initialized")
            if tensors is not None:
                raise Exception("ATerm improperly initialized")
            if index_key is not None:
                raise Exception("ATerm improperly initialized")
            self.scalar = copy(term.scalar)
            self.sums = [s.copy() for s in term.sums]
            self.tensors = [t.copy() for t in term.tensors]
            for d in term.deltas:
                self.tensors.append(tensor_from_delta(d))
            self.index_key = term.index_key
        else:
            if scalar is None:
                scalar = 1
            if sums is None or tensors is None:
                raise Exception("Improper initialization of ATerm")
            self.scalar = scalar
            self.sums = sums
            self.tensors = tensors
            self.index_key = index_key

    def __repr__(self):
        out = str(self.scalar)
        for ss in self.sums:
            out += str(ss)
        for tt in self.tensors:
            out += str(tt)
        return out

    def __mul__(self, other):
        if isinstance(other, Number):
            new = self.copy()
            new.scalar *= other
            return new
        elif isinstance(other, ATerm):
            sil1 = set(self.ilist())
            sil2 = set(other.ilist())
            if sil1.intersection(sil2):
                m = max([i.index for i in sil1])
                new = other._inc(m + 1)
            else:
                new = other
            scalar = self.scalar*new.scalar
            sums = self.sums + new.sums
            tensors = self.tensors + new.tensors
            return ATerm(
                scalar=scalar, sums=sums,
                tensors=tensors, index_key=self.index_key)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            new = self.copy()
            new.scalar *= other
            return new
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, ATerm):
            return (self.scalar == other.scalar
                    and set(self.sums) == set(other.sums)
                    and set(self.tensors) == set(other.tensors))
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, ATerm):
            if len(self.tensors) < len(other.tensors):
                return True
            elif len(self.tensors) == len(other.tensors):
                if len(self.sums) == len(other.sums):
                    if self.tensors == other.tensors:
                        return self.sums < other.sums
                    else:
                        return self.tensors < other.tensors
                else:
                    return len(self.sums) < len(other.sums)
            else:
                return False
        else:
            return NotImplemented

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def _inc(self, i):
        sums = [s._inc(i) for s in self.sums]
        tensors = [t._inc(i) for t in self.tensors]
        return ATerm(
            scalar=self.scalar, sums=sums,
            tensors=tensors, index_key=self.index_key)

    def _idx_map(self):
        if self.index_key is None:
            index_key = default_index_key
        else:
            index_key = self.index_key
        ilist = self.ilist()
        off = {}
        imap = {}
        for idx in ilist:
            s = idx.space
            if s in off:
                o = off[s]
                off[s] += 1
            else:
                o = 0
                off[s] = 1
            imap[idx] = index_key[s][o]
        return imap

    def _print_str(self, with_scalar=True):
        imap = self._idx_map()
        out = str(float(self.scalar)) if with_scalar else str()
        iis = str()
        for ss in self.sums:
            iis += imap[ss.idx]
        if iis:
            out += "\\sum_{" + iis + "}"
        for tt in self.tensors:
            out += tt._print_str(imap)
        return out

    def _einsum_str(self):
        imap = self._idx_map()
        sstr = str(float(self.scalar))
        fstr = str()
        istr = str()
        tstr = str()
        for tt in self.tensors:
            if not tt.name:
                fstr += tt._istr(imap)
            else:
                tstr += ", " + tt.name
                istr += tt._istr(imap) + ","
        return sstr + "*einsum('" + istr[:-1] + "->" + fstr + "'" + tstr + ")"

    def match(self, other):
        if isinstance(other, ATerm):
            TM1 = TermMap(self.sums, self.tensors)
            TM2 = TermMap(other.sums, other.tensors)
            return TM1 == TM2
        else:
            return NotImplemented

    def pmatch(self, other):
        if isinstance(other, ATerm):
            tlists = [t.sym.tlist for t in other.tensors]
            if len(other.tensors) != len(self.tensors):
                return None
            if len(self.sums) != len(other.sums):
                return None
            TM1 = TermMap(self.sums, self.tensors)
            for xs in product(*tlists):
                sign = 1
                for x in xs:
                    sign *= x[1]
                newtensors = [permute(t, x[0]) for t, x in zip(other.tensors, xs)]
                TM2 = TermMap(other.sums, newtensors)
                if TM1 == TM2:
                    return sign
            return None
        else:
            return NotImplemented

    def ilist(self):
        ilist = []
        for tt in self.tensors:
            itlst = tt.ilist()
            for ii in itlst:
                if ii not in ilist:
                    ilist.append(ii)
        for ss in self.sums:
            idx = ss.idx
            if idx not in ilist:
                ilist.append(idx)
        return ilist

    def nidx(self):
        return len(self.ilist())

    def sort_tensors(self):
        off = 0
        for i, tt in enumerate(self.tensors):
            if not tt.name:
                self.tensors[off], self.tensors[i] =\
                    self.tensors[i], self.tensors[off]
                off = off + 1

    def merge_external(self):
        # check for sorting of external indices
        ext = True
        for t in self.tensors:
            if not ext and not t.name:
                raise Exception(
                    "Cannot merge external indices in unsorted term")
            if t.name:
                ext = False

        # for the sorted term find the number of tensors
        num_ext = 0
        for t in self.tensors:
            if not t.name:
                num_ext = num_ext + 1

        if num_ext > 1:
            newtensors = [t.copy() for t in self.tensors[num_ext:]]
            ext_indices = []
            for t in self.tensors[:num_ext]:
                ext_indices += t.indices
            t_ext = Tensor(ext_indices, "")
            self.tensors = [t_ext] + newtensors

    def connected(self):
        ll = []
        rtensors = [t for t in self.tensors if (t.name and t.indices)]
        for s in self.sums:
            ll.append(s.idx)
        adj = []
        for idx in ll:
            xx = set()
            for i, t in enumerate(rtensors):
                if idx in t.indices:
                    xx.add(i)
            adj.append(xx)

        # If there are fewer than two tensors, there is no adjacency
        if not adj:
            return len(rtensors) < 2
        blue = set(adj[0])
        nb = len(blue)
        maxiter = 300000
        i = 0
        while i < maxiter:
            newtensors = []
            for b in blue:
                for ad in adj:
                    if b in ad:
                        for a in ad:
                            newtensors.append(a)
            blue = blue.union(set(newtensors))
            nb2 = len(blue)
            if nb2 == nb:
                break
            nb = nb2
            i += 1
        return len(set(blue)) == len(rtensors)

    def reducible(self):
        if not self.connected():
            return True
        for i, _ in enumerate(self.sums):
            new = self.copy()
            new._inc(1)
            sn = new.sums[i]
            i1 = sn.idx
            new.sums = list(filter(lambda s: s != sn, self.sums))
            m = 0
            for t in new.tensors:
                for ix in t.indices:
                    if ix == i1:
                        m += 1
            assert m == 2
            if not new.connected():
                return True

        return False

    def transpose(self, perm):
        self.merge_external()
        self.tensors[0].transpose(perm)

    def copy(self):
        newtensors = [t.copy() for t in self.tensors]
        newscalar = copy(self.scalar)
        newsums = [s.copy() for s in self.sums]
        return ATerm(
            scalar=newscalar, sums=newsums,
            tensors=newtensors, index_key=self.index_key)


class Expression(object):
    """Operator expression

    Attributes:
        terms (list): List of terms
        tthresh (float): Scalar thresholf for determining when terms are zero
    """
    def __init__(self, terms):
        self.terms = terms
        self.tthresh = 1e-15

    def resolve(self):
        for i in range(len(self.terms)):
            self.terms[i].resolve()

        # get rid of terms that are zero
        self.terms = list(
            filter(lambda x: abs(x.scalar) > self.tthresh, self.terms))

    def __repr__(self):
        out = str()
        for t in self.terms:
            out += str(t)
            out += " + "
        return out[:-3]

    def __add__(self, other):
        if isinstance(other, Expression):
            return Expression(self.terms + other.terms)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Expression):
            return self + -1*other
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Number):
            new = Expression([other*t for t in self.terms])
            return new
        elif isinstance(other, Expression):
            terms = [t1*t2 for t1, t2 in product(self.terms, other.terms)]
            return Expression(terms)
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Expression):
            # NOTE: This compares in fixed order with fixed indices
            if len(self.terms) != len(other.terms):
                return False
            for t1, t2 in zip(self.terms, other.terms):
                if t1 != t2:
                    return False
            return True
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    __rmul__ = __mul__

    def _print_str(self):
        out = str()
        for t in self.terms:
            sca = t.scalar
            num = abs(sca)
            sign = " + " if sca > 0 else " - "
            out += sign + str(num) + t._print_str(with_scalar=False) + "\n"
        return out[:-1]

    def are_operators(self):
        for i in range(len(self.terms)):
            if len(self.terms[i].operators) > 0:
                return True
        return False


class AExpression(object):
    """Abstract tensor expression

    Attributes:
        terms (list): list of Terms in the expression
        tthresh (float): Threshold for determining if a term is zero
    """
    def __init__(self, terms=None, Ex=None, simplify=True, sort=True):
        if terms is not None and Ex is None:
            self.terms = terms
        elif Ex is not None and terms is None:
            self.terms = [ATerm(term=t) for t in Ex.terms]
        else:
            self.terms = []
            simplify = False
            sort = False
        self.tthresh = 1e-15
        if simplify:
            self.simplify()
        if sort:
            self.sort()

    def simplify(self):
        # get rid of terms that are zero
        self.terms = list(
            filter(lambda x: abs(x.scalar) > self.tthresh, self.terms))

        # compress all symmetry-related terms
        newterms = []

        def test(x):
            return x[1] is not None

        while self.terms:
            t1 = self.terms[0]
            remaining = self.terms[1:]
            tm = list(filter(
                test, [(t, t1.pmatch(t), i) for i, t in enumerate(remaining)]))
            s = t1.scalar
            for t in tm:
                s += t[1]*t[0].scalar
            t1.scalar = s
            newterms.append(t1.copy())
            tmi = [t[2] for t in tm]
            indices = list(
                filter(lambda i: i not in tmi, range(0, len(remaining))))
            self.terms = [remaining[i] for i in indices]
        self.terms = newterms

        # get rid of terms that are zero after compression
        self.terms = list(
            filter(lambda x: abs(x.scalar) > self.tthresh, self.terms))

    def __repr__(self):
        return self._print_str()

    def __add__(self, other):
        if isinstance(other, AExpression):
            return AExpression(self.terms + other.terms)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, AExpression):
            return self + -1*other
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Number):
            new = AExpression(
                terms=[other*t for t in self.terms], simplify=False)
            return new
        elif isinstance(other, AExpression):
            terms = [t1*t2 for t1, t2 in product(self.terms, other.terms)]
            return AExpression(terms=terms)
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __eq__(self, other):
        if isinstance(other, AExpression):
            # NOTE: This compares in fixed order with fixed indices
            if len(self.terms) != len(other.terms):
                return False
            for t1, t2 in zip(self.terms, other.terms):
                if t1 != t2:
                    return False
            return True
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def _print_str(self):
        out = str()
        for t in self.terms:
            sca = t.scalar
            num = float(abs(sca))
            sign = " + " if sca > 0 else " - "
            out += sign + str(num) + t._print_str(with_scalar=False) + "\n"
        return out[:-1]

    def _print_einsum(self, lhs=None):
        X = lhs if lhs is not None else str()
        out = str()
        for t in self.terms[:-1]:
            out += X + " += " + t._einsum_str() + "\n"
        if self.terms:
            out += X + " += " + self.terms[-1]._einsum_str()
        return out

    def sort_tensors(self):
        for t in self.terms:
            t.sort_tensors()

    def sort(self):
        self.terms.sort()

    def connected(self):
        for t in self.terms:
            if not t.connected():
                return False
        return True

    def get_connected(self, simplify=True):
        newterms = [t for t in self.terms if t.connected()]
        return AExpression(terms=newterms, simplify=simplify)

    def pmatch(self, other):
        if isinstance(other, AExpression):
            if len(self.terms) != len(other.terms):
                return False
            for t1 in self.terms:
                matched = False
                for t2 in other.terms:
                    if t2.pmatch(t1):
                        matched = True
                        break
                if not matched:
                    return False
            return True
        else:
            return NotImplemented

    def transpose(self, perm):
        for t in self.terms:
            t.transpose(perm)
