from operator import Operator, Sigma, Delta
from expression import Term, Expression
import copy

def are_operators(e):
    for i in range(len(e.terms)):
        if len(e.terms[i].operators) > 0:
            return True

    return False

def is_occupied(s, occ):
    if occ is None:
        return 'o' in s.name
    else:
        return s in occ

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
        for i,oi in enumerate(temp.operators):
            for j,oj in enumerate(temp.operators):
                if j <= i or not oi.space == oj.space:
                    continue

                # if there is a non-zero contraction, do it
                if (is_occupied(oi.space, occ) and oi.ca and not oj.ca) or (
                    not is_occupied(oi.space, occ) and not oi.ca and oj.ca):
                    sign = 1 if (i - j + 1)%2 == 0 else -1
                    i1 = oi.index
                    i2 = oj.index
                    s1 = oi.space
                    s2 = oj.space
                    t1 = copy.deepcopy(temp)
                    t1.scalar = t1.scalar*sign
                    del(t1.operators[j])
                    del(t1.operators[i])
                    t1.deltas.append(Delta(i1,i2,s1,s2))

                    # append to output
                    to.append(t1)
                    break
        
    o = Expression(to)
    if are_operators(o):
        return(apply_wick(o))
    else:
        return o

