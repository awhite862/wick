from copy import deepcopy
from numbers import Number
from operator import Sigma, Delta, Operator, Tensor

#class AbIndex(object):
#    def __init__(self, ts, os, summed, space):
#        pass
#
#class TermMap(object):
#    def __init__(self, sums, tensors, operators, deltas):
#        assert(len(deltas) == 0)
#        sidx = [s.index for s in sums]
#        oidx = [o.index for o in operators]

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

            dindx = []
            for i in range(len(self.sums)):
                idx = self.sums[i].idx
                if i2 == idx:
                    dindx.append(i)
                    case = 3 if case == 1 else 2
                    break
                elif i1 == idx:
                    dindx.append(i)
                    case = 3 if case == 2 else 1
                    break;

            if len(dindx) > 0:
                dindx.sort()
                dindx = dindx[::-1]
                for j in dindx:
                    del self.sums[j]

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
            scalar = self.scalar*other.scalar
            sums = self.sums + other.sums
            tensors = self.tensors + other.tensors
            operators = self.operators + other.operators
            deltas = self.deltas + other.deltas
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

    #def match(self, other):
    #    if isinstance(other, Term):
    #        if len(self.deltas) > 0 or len(other.deltas) > 0:
    #            raise Exception("Cannot match terms with unresolved deltas!")

    #    else:
    #        return NotImplemented

class Expression(object):
    def __init__(self, terms):
        self.terms = terms

    def resolve(self):
        for i in range(len(self.terms)):
            self.terms[i].resolve()
    def __repr__(self):
        s = str()
        for t in self.terms:
           s = s + str(t)
           s = s + " + "

        return s[:-2]

    def __mul__(self, other):
        pass
