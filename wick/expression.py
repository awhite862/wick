from copy import deepcopy
from itertools import product
from numbers import Number
from .operator import Sigma, Delta, BOperator, FOperator, Tensor, permute, tensor_from_delta

class TermMap(object):
    """Map indicating the contraction pattern of a given tensor expression

    Attributes:
        data (set): set of tuples indicating a contraction pattern of a tensor expression
    """
    def __init__(self, sums, tensors, occ=None):
        self.data = set()
        for ti in tensors:
            colist = str()
            cvlist = str()
            cblist = str()
            for i,iidx in enumerate(ti.indices):
                fermion = iidx.fermion
                occupied = False if not fermion else iidx.is_occupied(occ=occ)
                for tj in tensors:
                    if tj == ti: continue
                    for j,jidx in enumerate(tj.indices):
                        if iidx == jidx:
                            tjname = tj.name if tj.name else "!"
                            cstr = str(i) + tjname + str(j)
                            if occupied: colist += cstr
                            elif fermion: cvlist += cstr
                            else: cblist += cstr
            tiname = ti.name if ti.name else "!"
            self.data.add((tiname,colist,cvlist,cblist))

    def __eq__(self, other):
        return self.data == other.data

def default_index_key():
    return {"occ" : "ijklmno", "vir" : "abcdefg", "nm" : "IJKLMNOP"}

class Term(object):
    """Term of operators

    Attributes:
        scalar (Number): scalar multiplying the term
        sums (list): list of sums over indices
        tensors (list): list of tensors
        operators (list): list of creation/anihillation operators
        deltas (list): list of delta functions
    """
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

class ATerm(object):
    """Abstract term

    Attributes:
        scalar (Number): scalar constant multiplying the term
        sums (list): list of Sums in the term
        tensors (list): list of Tensors
    """
    def __init__(self, scalar=None, sums=None, tensors=None, term=None):
        if term is not None:
            #assert(len(term.deltas) == 0)
            assert(len(term.operators) == 0)
            if scalar is not None:
                raise Exception("ATerm improperly initialized")
            if sums is not None:
                raise Exception("ATerm improperly initialized")
            if tensors is not None:
                raise Exception("ATerm improperly initialized")
            self.scalar = deepcopy(term.scalar)
            self.sums = deepcopy(term.sums)
            self.tensors = deepcopy(term.tensors)
            for d in term.deltas:
                self.tensors.append(tensor_from_delta(d))
        else:
            if scalar is None: scalar = 1
            if sums is None or tensors is None:
                raise Exception("Improper initialization of ATerm")
            self.scalar = scalar
            self.sums = sums
            self.tensors = tensors

    def __repr__(self):
        out = str(self.scalar)
        for ss in self.sums:
            out += str(ss)
        for tt in self.tensors:
            out += str(tt)
        return out

    def __mul__(self, other):
        if isinstance(other, Number):
            new = deepcopy(self)
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
            return ATerm(scalar=scalar, sums=sums, tensors=tensors)
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
        if isinstance(other, ATerm):
            return self.scalar == other.scalar \
                    and set(self.sums) == set(other.sums) \
                    and set(self.tensors) == set(other.tensors)
        else:
            return NotImplemented

    def __neq__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, ATerm):
            if len(self.tensors) < len(other.tensors): return True
            elif len(self.tensors) == len(other.tensors):
                return len(self.sums) < len(other.sums)
            else:
                return False
        else:
            return NotImplemented

    def _inc(self, i):
        sums = [s._inc(i) for s in self.sums]
        tensors = [t._inc(i) for t in self.tensors]
        return Term(scalar=self.scalar, sums=sums, tensors=tensors)

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
        out = str(float(self.scalar)) if with_scalar else str()
        iis = str()
        for ss in self.sums:
            iis += imap[ss.idx]
        if iis:
            out += "\sum_{" + iis + "}"
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
            if len(other.tensors) != len(self.tensors): return None
            if len(self.sums) != len(other.sums): return None
            TM1 = TermMap(self.sums, self.tensors)
            for xs in product(*tlists):
                sign = 1
                for x in xs: sign *= x[1]
                newtensors = [permute(t,x[0]) for t,x in zip(other.tensors, xs)]
                TM2 = TermMap(other.sums, newtensors)
                if TM1 == TM2: 
                    return sign
            return None
        else: return NotImplemented

    def ilist(self):
        ilist = []
        for tt in self.tensors:
            itlst = tt.ilist()
            for ii in itlst:
                if ii not in ilist: ilist.append(ii)
        for ss in self.sums:
            idx = ss.idx
            if idx not in ilist: ilist.append(idx)
        return ilist

    def nidx(self):
        return len(self.ilist())

    def sort_tensors(self):
        off = 0
        for i,tt in enumerate(self.tensors):
            if not tt.name:
                self.tensors[off],self.tensors[i] = self.tensors[i], self.tensors[off]
                off = off + 1

    def merge_external(self):
        # check for sorting of external indices
        ext = True
        for t in self.tensors:
            if ext == False and not t.name:
                raise Exception("Cannot merge external indices in unsorted term")
            if t.name: ext = False

        # for the sorted term find the number of tensors
        num_ext = 0
        for t in self.tensors:
            if not t.name: num_ext = num_ext + 1

        # check for symmetry in external indices

        if num_ext < 2: pass
        else:
            newtensors = deepcopy(self.tensors[num_ext:])
            ext_indices = []
            for t in self.tensors[:num_ext]:
                ext_indices += t.indices
            t_ext = Tensor(ext_indices, "")
            self.tensors = [t_ext] + newtensors

    def connected(self):
        ll = []
        rtensors = [t for t in self.tensors if t.name]
        for s in self.sums:
            ll.append(s.idx)
        adj = []
        for idx in ll:
            xx = set()
            for i,t in enumerate(rtensors):
                if idx in t.indices:
                    xx.add(i)
            adj.append(xx)

        # If there are fewer than two tensors, there is no adjacency
        if not adj: return (len(rtensors) < 2)
        blue = set(adj[0])
        nb = len(blue)
        maxiter = 300000
        while i < maxiter:
            newtensors = []
            for b in blue:
                for ad in adj:
                    if b in ad:
                        for a in ad: newtensors.append(a)
            blue = blue.union(set(newtensors))
            nb2 = len(blue)
            if nb2 == nb: break
            nb = nb2
            i += 1
        return len(set(blue)) == len(rtensors)

    def transpose(self, perm):
        self.merge_external()
        self.tensors[0].transpose(perm)

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
        self.terms = list(filter(lambda x: abs(x.scalar) > self.tthresh, self.terms))

    def __repr__(self):
        out = str()
        for t in self.terms:
           out += str(t)
           out += " + "
        return out[:-2]

    def __add__(self, other):
        if isinstance(other, Expression):
            return Expression(self.terms + other.terms)
        else: return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Expression):
            return self + -1*other

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
        return self._print_str()

    def __add__(self, other):
        if isinstance(other, AExpression):
            return AExpression(self.terms + other.terms)
        else: return NotImplemented

    def __sub__(self, other):
        if isinstance(other, AExpression):
            return self + -1*other

    def __mul__(self, other):
        if isinstance(other, Number):
            new = Expression([other*t for t in self.terms])
            return new
        elif isinstance(other, AExpression):
            terms = [t1*t2 for t1,t2 in product(self.terms, other.terms)]
            return Expression(terms)
        else: return NotImplemented

    __rmul__ = __mul__

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
            if not t.connected(): return False
        return True

    def get_connected(self):
        newterms = [t for t in self.terms if t.connected()]
        return AExpression(terms=newterms)

    def transpose(self, perm):
        for t in self.terms:
            t.transpose(perm)
