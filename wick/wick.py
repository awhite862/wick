from .operator import BOperator, FOperator, Sigma, Delta
from .expression import Term, Expression
import copy

def valid_contraction(o1, o2, occ=None):
    if isinstance(o1, FOperator) and isinstance(o2, FOperator):
         if o1.idx.space != o2.idx.space:
             return False
         if (o1.idx.is_occupied(occ=occ) and o1.ca and not o2.ca) or (
             not o1.idx.is_occupied(occ=occ) and not o1.ca and o2.ca):
             return True
         return False
    elif isinstance(o1, BOperator) and isinstance(o2, BOperator):
         if o1.idx.space != o2.idx.space:
             return False
         if (not o1.ca and o2.ca):
             return True
         return False
    elif type(o1) is not type(o1): return False
    else:
        return True

def pair_list(lst,occ=None):
    n = len(lst)
    assert(n%2 == 0)
    if n < 2:
        return []
    if n == 2:
        if valid_contraction(lst[0],lst[1],occ=occ):
            return [[(lst[0],lst[1])],]
        else:
            return []
    else:
        plist = []
        for i,x in enumerate(lst[1:]):
            if valid_contraction(lst[0], x):
                p1 = [(lst[0],x),]
                remainder = pair_list(lst[1:i+1] + lst[i+2:])
                for r in remainder:
                    plist.append(p1 + r)
        return plist

def find_pair(i, ipairs):
    for p in ipairs:
        if p[0] == i or p[1] ==i:
            return p

def get_sign(ipairs):
    ncross = 0
    for p in ipairs:
        i,j = p
        for x1 in range(i+1,j):
            p1 = find_pair(x1, ipairs)
            if p1 is None: continue
            x2 = p1[0] if p1[1] == x1 else p1[1]
            if x2 > j or x2 < i: ncross += 1

    assert(ncross%2 == 0)
    ncross = ncross//2
    return 1.0 if ncross%2 == 0 else -1.0

def apply_wick(e, occ=None):
    to = []
    # loop over terms
    for temp in e.terms:
        # if there is an odd number of operators, then we are done
        if len(temp.operators)%2 != 0:
            continue
        if len(temp.operators) == 0:
            to.append(copy.deepcopy(temp))
            continue
        # loop to find a contraction
        plist = pair_list(temp.operators)
        for pairs in plist:
            good = bool(pairs)
            ipairs = []
            deltas = []
            for p in pairs:
                oi,oj = p
                if oi.idx.space != oj.idx.space:
                    good = False
                    break
                if not oi.idx.fermion:
                    i = temp.operators.index(oi)
                    j = temp.operators.index(oj)
                    i1 = oi.idx
                    i2 = oj.idx
                    deltas.append(Delta(i1,i2))
                elif (oi.idx.is_occupied(occ=occ) and oi.ca and not oj.ca) or (
                    not oi.idx.is_occupied(occ=occ) and not oi.ca and oj.ca):
                    i = temp.operators.index(oi)
                    j = temp.operators.index(oj)
                    ipairs.append((i,j))
                    i1 = oi.idx
                    i2 = oj.idx
                    deltas.append(Delta(i1,i2))
                else:
                    good = False
                    break
            # append to output
            if good:
                sign = get_sign(ipairs)
                t1 = Term(sign*temp.scalar,
                        copy.deepcopy(temp.sums),
                        copy.deepcopy(temp.tensors),
                        [],
                        deltas + copy.deepcopy(temp.deltas))
                to.append(t1)
        
    o = Expression(to)
    if o.are_operators():
        raise Exception("Application of Wick's theorem has failed!")
    else:
        return o

