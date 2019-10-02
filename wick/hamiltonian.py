from itertools import product
from .index import Idx
from .operator import BOperator, FOperator, TensorSym, Tensor, Sigma
from .expression import Term, Expression

def is_normal_ordered(operators, occ):
    fa = None
    for i,op in enumerate(operators):
        if fa is None and (not op.qp_creation(occ)): fa = i
        if fa is not None and op.qp_creation(occ): return False
    return True

def normal_ordered(operators,occ=None,sign=1.0):
    if is_normal_ordered(operators, occ):
        return (operators,sign)
    fa = None
    swap = None
    for i,op in enumerate(operators):
        if fa is None and (not op.qp_creation(occ)): fa = i
        if fa is not None and op.qp_creation(occ):
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
            operators = [FOperator(I1, True), FOperator(I2, False)]
            nsign = 1.0
            if norder:
                operators,nsign = normal_ordered([FOperator(I1, True), FOperator(I2, False)])
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
                xx = list(filter(lambda x: x,[s3 == s for s in [s1,s2]]))
                I3 = Idx(len(xx),s3)
                for s4 in spaces:
                    xx = list(filter(lambda x: x,[s4 == s for s in [s1,s2,s3]]))
                    I4 = Idx(len(xx),s4)
                    operators = [FOperator(I1, True), FOperator(I2, True), FOperator(I4,False), FOperator(I3,False)]
                    nsign = 1.0
                    if norder:
                        operators,nsign = normal_ordered(operators)
                    t = Term(nsign*fac, [Sigma(I1),Sigma(I2),Sigma(I3),Sigma(I4)],
                            [Tensor([I1,I2,I3,I4],name,sym=sym)],
                            operators,
                            [])
                    terms.append(t)
    return Expression(terms)

def one_p(name, space="nm"):
    I1 = Idx(0, space, fermion=False)
    tc = Term(1.0, [Sigma(I1)],
            [Tensor([I1],name)],
            [BOperator(I1, True)],[])
    ta = Term(1.0, [Sigma(I1)],
            [Tensor([I1],name)],
            [BOperator(I1, False)],[])
    terms = [tc,ta]
    return Expression(terms)

def two_p(name, space="nm",diag=True):
    I1 = Idx(0, space, fermion=False)
    if diag:
        t1 = Term(1.0, [Sigma(I1)],
                [Tensor([I1],name)],
                [BOperator(I1, True),BOperator(I1,False)],[])
    else:
        I2 = Idx(1, space, fermion=False)
        t1 = Term(1.0, [Sigma(I1),Sigma(I2)],
                [Tensor([I1,I2],name)],
                [BOperator(I1, True),BOperator(I2, False)],[])
    return Expression([t1])

def ep11(name, fspaces, bspaces, norder=False):
    terms = []
    for sb in bspaces:
        I1 = Idx(0, sb, fermion=False)
        for s1 in fspaces:
            p1 = Idx(0, s1)
            for s2 in fspaces:
                i = 0 if s2 != s1 else 1
                p2 = Idx(i,s2)
                operators = [FOperator(p1, True), FOperator(p2, False)]
                nsign = 1.0
                if norder:
                    operators,nsign = normal_ordered([FOperator(p1, True), FOperator(p2, False)])
                tc = Term(nsign, [Sigma(I1),Sigma(p1),Sigma(p2)],
                        [Tensor([I1,p1,p2],name)],
                        [BOperator(I1, True)] + operators,
                        [])
                ta = Term(nsign, [Sigma(I1),Sigma(p1),Sigma(p2)],
                        [Tensor([I1,p1,p2],name)],
                        [BOperator(I1, False)] + operators,
                        [])
                terms.append(ta)
                terms.append(tc)
    return Expression(terms)

def E0(name):
    """
    Return a zero-rank constant tensor
    """
    return Expression([Term(1.0,[], [Tensor([], name)], [], [])])

def E1(name, ospaces, vspaces):
    """
    Return the tensor representation of a Fermion excitation operator

    name (string): name of the tensor
    ospaces (list): list of occupied spaces
    vspaces (list): list of virtual spaces
    """
    terms = []
    for os in ospaces:
        for vs in vspaces:
            i = Idx(0, os)
            a = Idx(0, vs)
            e1 = Term(1.0,
                [Sigma(i), Sigma(a)],
                [Tensor([a, i], name)],
                [FOperator(a, True), FOperator(i, False)],
                [])
            terms.append(e1)
    return Expression(terms)

def E2(name, ospaces, vspaces):
    """
    Return the tensor representation of a Fermion excitation operator

    name (string): name of the tensor
    ospaces (list): list of occupied spaces
    vspaces (list): list of virtual spaces
    """
    terms = []
    sym = get_sym(True)
    for i1,o1 in enumerate(ospaces):
        for o2 in ospaces[i1:]:
            for j1,v1 in enumerate(vspaces):
                for v2 in vspaces[j1:]:
                    i = Idx(0, o1)
                    a = Idx(0, v1)
                    j = Idx(1, o2)
                    b = Idx(1, v2)
                    e2 = Term(0.25,
                        [Sigma(i), Sigma(a), Sigma(j), Sigma(b)],
                        [Tensor([a, b, i, j], name, sym=sym)],
                        [FOperator(a, True), FOperator(i, False),
                            FOperator(b, True), FOperator(j, False)],
                        [])
                    terms.append(e2)
    return Expression(terms)

def P1(name, spaces):
    """
    Return the tensor representation of a Boson excitation operator

    name (string): name of the tensor
    spaces (list): list of spaces
    """
    terms = []
    for s in spaces:
        I = Idx(0, s, fermion=False)
        e1 = Term(1.0,
            [Sigma(I)],
            [Tensor([I], name)],
            [BOperator(I, True)],
            [])
        terms.append(e1)
    return Expression(terms)

def P2(name, spaces):
    """
    Return the tensor representation of a Boson double-excitation operator

    name (string): name of the tensor
    spaces (list): list of spaces
    """
    terms = []
    sym = TensorSym([(0,1),(1,0)], [1.0,1.0])
    for s1 in spaces:
        for s2 in spaces:
            I = Idx(0, s1, fermion=False)
            i = 0 if s1 == s2 else 1
            J = Idx(i, s2, fermion=False)
            e2 = Term(1.0,
                [Sigma(I),Sigma(J)],
                [Tensor([I,J], name, sym=sym)],
                [BOperator(I, True),BOperator(J, True)],
                [])
            terms.append(e2)
    return Expression(terms)

def EPS1(name, bspaces, ospaces, vspaces):
    """
    Return the tensor representation of a coupled
    Fermion-Boson excitation operator

    name (string): name of the tensor
    bspaces (list): list of Boson spaces
    ospaces (list): list of occupied spaces
    vspaces (list): list of virtual spaces
    """
    terms = []
    for bs in bspaces:
        for os in ospaces:
            for vs in vspaces:
                I = Idx(0, bs, fermion=False)
                i = Idx(0, os)
                a = Idx(0, vs)
                e1 = Term(1.0,
                    [Sigma(I), Sigma(i), Sigma(a)],
                    [Tensor([I, a, i], name)],
                    [BOperator(I, True), FOperator(a, True), FOperator(i, False)],
                    [])
                terms.append(e1)
    return Expression(terms)

def projE0():
    """
    Return a projector onto the vacuum.
    """
    return Expression([Term(1.0, [], [Tensor([], "")], [], [])])

def projE1(ospace, vspace):
    """
    Return left-projector onto a space of single excitations

    ospace (str): occupied space
    vspace (str): virtual space
    """
    i = Idx(0, ospace)
    a = Idx(0, vspace)
    operators = [FOperator(i,True), FOperator(a,False)]
    return Expression([Term(1.0, [], [Tensor([a,i],"")], operators, [])])

def projE2(o1, v1, o2, v2):
    """
    Return left-projector onto a space of double excitations

    o1 (str): 1st occupied space
    v1 (str): 1st virtual space
    o2 (str): 2nd occupied space
    v2 (str): 2nd virtual space
    """
    i = Idx(0, o1)
    a = Idx(0, v1)
    x = 1 if o2 == o1 else 0
    y = 1 if v2 == v1 else 0
    j = Idx(x, o1)
    b = Idx(y, v1)
    operators = [FOperator(i,True), FOperator(a,False), FOperator(j,True), FOperator(b,False)]
    return Expression([Term(1.0, [], [Tensor([a,b,i,j],"")], operators, [])])

def projP1(space):
    """
    Return projection onto single Boson space
    """
    I = Idx(0, space, fermion=False)
    return Expression([Term(1.0, [], [Tensor([I],"")], [BOperator(I, False)], [])])

def commute(A, B):
    """ Return the commutator of two operators"""
    return A*B - B*A
