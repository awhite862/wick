from .operator import Operator, Sigma, Delta
from .expression import Term, Expression
import copy

def are_operators(e):
    for i in range(len(e.terms)):
        if len(e.terms[i].operators) > 0:
            return True
    return False

def is_occupied(s, occ):
    if occ is None:
        return 'o' in s
    else:
        return s in occ

def qp_creation(op, occ=None):
    if (not is_occupied(op.idx.space, occ)) and op.ca:
        return True
    elif (is_occupied(op.idx.space, occ)) and not op.ca:
        return True
    else:
        return False

def qp_anihilation(op, occ=None):
    return not qp_creation(op, occ=occ)

def pair_list(lst):
    n = len(lst)
    assert(n%2 == 0)
    if n < 2:
        return [[],]
    if n == 2:
        return [[(lst[0],lst[1])],]
    else:
        plist = []
        for i,x in enumerate(lst[1:]):
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
            t1 = copy.deepcopy(temp)
            t1.operators = []
            good = True
            ipairs = []
            for p in pairs:
                oi,oj = p
                if oi.idx.space != oj.idx.space:
                    good = False
                    break
                if (is_occupied(oi.idx.space, occ) and oi.ca and not oj.ca) or (
                    not is_occupied(oi.idx.space, occ) and not oi.ca and oj.ca):
                    i = temp.operators.index(oi)
                    j = temp.operators.index(oj)
                    ipairs.append((i,j))
                    i1 = oi.idx
                    i2 = oj.idx
                    t1.deltas.append(Delta(i1,i2))
                else:
                    good = False
                    break
            # append to output
            if good:
                sign = get_sign(ipairs)
                t1.scalar = sign*t1.scalar
                to.append(t1)
        
    o = Expression(to)
    if are_operators(o):
        raise Exception("Application of Wick's theorem has failed!")
    else:
        return o

