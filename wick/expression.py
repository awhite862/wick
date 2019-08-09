from copy import deepcopy
from itertools import product
from numbers import Number
from .operator import Sigma, Delta, Operator, Tensor, permute

# TODO: move this
def is_occupied(s, occ):
    if occ is None:
        return 'o' in s
    else:
        return s in occ

# TODO use only for more abstract class
class TermMap(object):

    class AbIdx(object):
        def __init__(self, idx, ts, os, summed):
            self.idx = idx
            self.ts = ts
            self.os = os
            self.summed = summed

        def __eq__(self, other):
            first = self.idx.space == other.idx.space and self.summed == other.summed
            if not first: return False
            if not self.os == other.os: return False
            for t,i in self.ts:
                found = False
                for s,j in other.ts:
                    if t.name == s.name and i == j:
                        found = True
                        break
                if not found: return False
            return True

    def __init__(self, sums, tensors, operators, deltas, occ=None):
        assert(len(deltas) == 0)
        assert(len(operators) == 0)
        self.data = set()
        for ti in tensors:
            colist = str()
            cvlist = str()
            for i,iidx in enumerate(ti.indices):
                occupied = is_occupied(iidx.space, occ)
                for tj in tensors:
                    if tj == ti: continue
                    for j,jidx in enumerate(tj.indices):
                        if iidx == jidx:
                            cstr = str(i) + tj.name + str(j)
                            if occupied: colist += cstr
                            else: cvlist += cstr
            self.data.add((ti.name,colist,cvlist))

    def __eq__(self, other):
        return self.data == other.data

def default_index_key():
    return {"occ" : "ijklmno", "vir" : "abcdefg"}

class Term(object):
    def __init__(self, scalar, sums, tensors, operators, deltas):
        self.scalar = scalar
        self.sums = sums
        self.tensors = tensors
        self.operators = operators
        self.deltas = deltas

    def resolve(self):
        dnew = []

        # get unique deltas
        self.deltas = list(set(self.deltas))
        
        # loop over deltas
        for dd in self.deltas:
            i2 = dd.i2
            i1 = dd.i1
            assert(i1.space == i2.space)

            ## Cases ##
            # 0 sums over neither index
            # 1 sums over 1st index
            # 2 sums over 2nd index
            # 3 sums over both indices
            case = 0

            dindx = -1 # index of sum to delete
            for i,s in enumerate(self.sums):
                idx = s.idx
                if i2 == idx:
                    dindx = i
                    case = 3 if case == 1 else 2
                elif i1 == idx:
                    case = 3 if case == 2 else 1
                    if case != 3: dindx = i

            if dindx >= 0:
                del self.sums[dindx]

            for tt in self.tensors:
                for k in range(len(tt.indices)):
                    if case == 1:
                        if tt.indices[k] == i1:
                            tt.indices[k] = i2
                    else:
                        if tt.indices[k] == i2:
                            tt.indices[k] = i1

            for oo in self.operators:
                if case == 1:
                    if oo.index == i1:
                        oo.index = i2
                else:
                    if oo.index == i2:
                        oo.index = i1

            if case == 0 and i1 != i2:
                dnew.append(dd)

        self.deltas = dnew

    def __repr__(self):
        s = str(self.scalar)
        for ss in self.sums:
            s = s + str(ss)
        for dd in self.deltas:
            s = s + str(dd)
        for tt in self.tensors:
            s = s + str(tt)
        for oo in self.operators:
            s = s + str(oo)
        return s


    def __mul__(self, other):
        if isinstance(other, Number):
            new = deepcopy(self)
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
            return Term(scalar, sums, tensors, operators, deltas)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            new = deepcopy(self)
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

    def __neq__(self, other):
        return not self.__eq__(other)

    def _inc(self, i):
        sums = [s._inc(i) for s in self.sums]
        tensors = [t._inc(i) for t in self.tensors]
        operators = [o._inc(i) for o in self.operators]
        deltas = [d._inc(i) for d in self.deltas]
        return Term(self.scalar, sums, tensors, operators, deltas)

    def _idx_map(self, indices=None):
        if indices is None:
            indices = default_index_key()
        ilist = self.ilist()
        off = {}
        imap = {}
        for idx in ilist:
            n,s = idx.index,idx.space
            if s in off:
                o = off[s]
                off[s] += 1
            else:
                o = 0
                off[s] = 1
            imap[idx] = indices[s][o]
        return imap

    def _print_str(self,with_scalar=True):
        imap = self._idx_map()
        s = str(self.scalar) if with_scalar else str()
        for ss in self.sums:
            s += ss._print_str(imap)
        for dd in self.deltas:
            s += dd._print_str(imap)
        for tt in self.tensors:
            s += tt._print_str(imap)
        for oo in self.operators:
            s += oo._print_str(imap)
        return s

    def match(self, other):
        if isinstance(other, Term):
            if len(self.deltas) > 0 or len(other.deltas) > 0:
                raise Exception("Cannot match terms with unresolved deltas!")
            TM1 = TermMap(self.sums, self.tensors, self.operators, self.deltas)
            TM2 = TermMap(other.sums, other.tensors, other.operators, other.deltas)
            return TM1 == TM2
        else:
            return NotImplemented

    def pmatch(self, other):
        if isinstance(other, Term):
            tlists = [t.sym.tlist for t in other.tensors]
            TM1 = TermMap(self.sums, self.tensors, self.operators, self.deltas)
            for xs in product(*tlists):
                sign = 1.0
                for x in xs: sign *= x[1]
                newtensors = [permute(t,x[0]) for t,x in zip(other.tensors, xs)]
                TM2 = TermMap(other.sums, newtensors, other.operators, other.deltas)
                if TM1 == TM2: 
                    return sign
            return None
        else: return NotImplemented

    def ilist(self):
        ilist = []
        for oo in self.operators:
            idx = oo.idx
            if idx not in ilist: ilist.append(idx)
        for tt in self.tensors:
            itlst = tt.ilist()
            for ii in itlst:
                if ii not in ilist: ilist.append(ii)
        for ss in self.sums:
            idx = ss.idx
            if idx not in ilist: ilist.append(idx)
        for dd in self.deltas:
            ii1 = dd.i1
            ii2 = dd.i2
            if ii1 not in ilist: ilist.append(ii1)
            if ii2 not in ilist: ilist.append(ii2)
        return ilist

class Expression(object):
    def __init__(self, terms):
        self.terms = terms
        self.tthresh = 1e-15

    def resolve(self):
        for i in range(len(self.terms)):
            self.terms[i].resolve()

        # get rid of terms that are zero
        self.terms = list(filter(lambda x: abs(x.scalar) > self.tthresh, self.terms))

        # compress all symmetry-related terms
        newterms = []
        while self.terms:
            t1 = self.terms[0]
            tm = list(filter(lambda x: x[1] is not None,[(t,t1.pmatch(t)) for t in self.terms[1:]]))
            s = t1.scalar
            for t in tm: s += t[1]*t[0].scalar
            t1.scalar = s
            newterms.append(deepcopy(t1))
            tm = [t[0] for t in tm]
            self.terms = list(filter(lambda x: x not in tm, self.terms[1:]))
        self.terms = newterms

        # get rid of terms that are zero after compression
        self.terms = list(filter(lambda x: abs(x.scalar) > self.tthresh, self.terms))

    def __repr__(self):
        s = str()
        for t in self.terms:
           s = s + str(t)
           s = s + " + "

        return s[:-2]

    def __add__(self, other):
        if isinstance(other, Expression):
            return Expression(self.terms + other.terms)
        else: return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Expression):
            return self + -1.0*other

    def __mul__(self, other):
        if isinstance(other, Number):
            new = Expression([other*t for t in self.terms])
            return new
        elif isinstance(other, Expression):
            terms = [t1*t2 for t1,t2 in product(self.terms, other.terms)]
            return Expression(terms)
        else: return NotImplemented

    __rmul__ = __mul__

    def _print_str(self):
        s = str()
        for t in self.terms:
            sca = t.scalar
            num = abs(sca)
            sign = " + " if sca > 0 else " - "
            s += sign + str(num) + t._print_str(with_scalar=False) + "\n"

        return s[:-1]
