from itertools import product
from .index import Idx
from .operator import Operator, TensorSym, Tensor, Sigma
from .expression import Term, Expression
from .wick import qp_creation

def is_normal_ordered(operators, occ):
    fa = None
    for i,op in enumerate(operators):
        if fa is None and (not qp_creation(op, occ)): fa = i
        if fa is not None and qp_creation(op, occ): return False
    return True

def normal_ordered(operators,occ=None,sign=1.0):
    if is_normal_ordered(operators, occ):
        return (operators,sign)
    fa = None
    swap = None
    for i,op in enumerate(operators):
        if fa is None and (not qp_creation(op, occ)): fa = i
        if fa is not None and qp_creation(op, occ):
            swap = i
            break
    assert(swap is not None)
    newops = operators[:fa] + [operators[swap]] + operators[fa:swap] + operators[swap+1:]
    newsign = 1.0 if len(operators[fa:swap])%2 == 0 else -1.0
    sign = sign*newsign
    return normal_ordered(newops,sign=sign)

def one_e(name, spaces, norder=False):
    terms = []
    for s1 in spaces:
        I1 = Idx(0, s1)
        for s2 in spaces:
            i = 0 if s2 != s1 else 1
            I2 = Idx(i,s2)
            operators = [Operator(I1, True), Operator(I2, False)]
            nsign = 1.0
            if norder:
                operators,nsign = normal_ordered([Operator(I1, True), Operator(I2, False)])
            t = Term(nsign, [Sigma(I1),Sigma(I2)],
                    [Tensor([I1,I2],name)],
                    operators,
                    [])
            terms.append(t)
    return Expression(terms)

def get_sym(anti):
    if anti:
        return TensorSym([(0,1,2,3),(1,0,2,3),(0,1,3,2),(1,0,3,2)],
                [1.0, -1.0, -1.0, 1.0])
    else:
        return TensorSym([(0,1,2,3),(1,0,3,2)],[1.0,1.0])

def two_e(name, spaces, anti=True, norder=False):
    terms = []
    sym = get_sym(anti)
    fac = 0.25 if anti else 0.5
    for s1 in spaces:
        I1 = Idx(0, s1)
        for s2 in spaces:
            i = 0 if s2 != s1 else 1
            I2 = Idx(i,s2)
            for s3 in spaces:
                xx = filter(lambda x: x,[s3 == s for s in [s1,s2]])
                I3 = Idx(len(xx),s3)
                for s4 in spaces:
                    xx = filter(lambda x: x,[s4 == s for s in [s1,s2,s3]])
                    I4 = Idx(len(xx),s4)
                    operators = [Operator(I1, True), Operator(I2, True), Operator(I4,False), Operator(I3,False)]
                    nsign = 1.0
                    if norder:
                        operators,nsign = normal_ordered(operators)
                    t = Term(nsign*fac, [Sigma(I1),Sigma(I2),Sigma(I3),Sigma(I4)],
                            [Tensor([I1,I2,I3,I4],name,sym=sym)],
                            operators,
                            [])
                    terms.append(t)
    return Expression(terms)
