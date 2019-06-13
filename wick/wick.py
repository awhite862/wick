from operator import Operator, Sigma, Delta
from expression import Term, Expression
import copy

def are_operators(e):
    for i in range(len(e.terms)):
        if len(e.terms[i].operators) > 0:
            return True

    return False

def apply_wick(e):
    to = []
    # loop over terms
    for temp in e.terms:
        # if there is an odd number of operators, then we are done
        if len(temp.operators)%2 != 0:
            continue

        # make list of all possible contractions
        for i in range(len(temp.operators)):
            for j in range(len(temp.operators)):
                if j <= i or not temp.operators[i].space == \
                    temp.operators[j].space:
                    continue

                oi = temp.operators[i]
                oj = temp.operators[j]

                # if there is a non-zero contraction, add it
                if ('o' in oi.space.name and oi.ca and not oj.ca) or (
                    'v' in oi.space.name and not oi.ca and oj.ca):
                    sign = 1 if (i - j + 1)%2 == 0 else -1
                    i1 = temp.operators[i].index
                    i2 = temp.operators[j].index
                    s1 = temp.operators[i].space
                    s2 = temp.operators[j].space
                    t1 = copy.deepcopy(temp)
                    t1.scalar = t1.scalar*sign
                    del(t1.operators[j])
                    del(t1.operators[i])
                    t1.deltas.append(Delta(i1,i2,s1,s2))

                    # append to output
                    to.append(t1)
        
    o = Expression(to)
    if are_operators(o):
        return(apply_wick(o))
    else:
        return o

