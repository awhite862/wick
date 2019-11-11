from itertools import product
from .index import Idx
from .operator import BOperator, FOperator, TensorSym, Tensor, Sigma, normal_ordered
from .expression import Term, Expression

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

def get_sym_ip2():
    return TensorSym([(0,1,2),(0,2,1)],
                [1.0, -1.0])

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

def one_p(name, space="nm", name2 = None):
    if name2 is None: name2 = name
    I1 = Idx(0, space, fermion=False)
    tc = Term(1.0, [Sigma(I1)],
            [Tensor([I1],name2)],
            [BOperator(I1, True)],[])
    ta = Term(1.0, [Sigma(I1)],
            [Tensor([I1],name)],
            [BOperator(I1, False)],[])
    terms = [tc,ta]
    return Expression(terms)

def two_p(name, space="nm"):
    I1 = Idx(0, space, fermion=False)
    I2 = Idx(1, space, fermion=False)
    t1 = Term(1.0, [Sigma(I1),Sigma(I2)],
            [Tensor([I1,I2],name)],
            [BOperator(I1, True),BOperator(I2, False)],[])
    return Expression([t1])

def ep11(name, fspaces, bspaces, norder=False, name2=None):
    terms = []
    if name2 is None: name2 = name
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
                        [Tensor([I1,p1,p2],name2)],
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
    Project onto the vacuum
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
                        [FOperator(a, True), FOperator(b, True),
                            FOperator(j, False), FOperator(i, False)],
                        [])
                    terms.append(e2)
    return Expression(terms)

def Eip1(name, ospaces):
    """
    Return the tensor representation of a Fermion ionization

    name (string): name of the tensor
    ospaces (list): list of occupied spaces
    """
    terms = []
    for os in ospaces:
        i = Idx(0, os)
        e1 = Term(1.0,
            [Sigma(i)],
            [Tensor([i], name)],
            [FOperator(i, False)],
            [])
        terms.append(e1)
    return Expression(terms)

def Eip2(name, ospaces, vspaces):
    """
    Return the tensor representation of a Fermion ip (trion)

    name (string): name of the tensor
    ospaces (list): list of occupied spaces
    vspaces (list): list of virtual spaces
    """
    terms = []
    sym = get_sym_ip2()
    for i1,o1 in enumerate(ospaces):
        for o2 in ospaces[i1:]:
            for j1,v1 in enumerate(vspaces):
                i = Idx(0, o1)
                a = Idx(0, v1)
                j = Idx(1, o2)
                e2 = Term(0.5,
                    [Sigma(i), Sigma(a), Sigma(j)],
                    [Tensor([a, i, j], name, sym=sym)],
                    [FOperator(a, True), FOperator(j, False), FOperator(i, False)],
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
            i = 1 if s1 == s2 else 0
            J = Idx(i, s2, fermion=False)
            e2 = Term(0.5,
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

def EPS2(name, bspaces, ospaces, vspaces):
    """
    Return the tensor representation of a coupled
    Fermion-double Boson excitation operator

    name (string): name of the tensor
    bspaces (list): list of Boson spaces
    ospaces (list): list of occupied spaces
    vspaces (list): list of virtual spaces
    """
    terms = []
    sym = TensorSym([(0,1,2,3),(1,0,2,3)], [1.0,1.0])
    for b1 in bspaces:
        for b2 in bspaces:
            for os in ospaces:
                for vs in vspaces:
                    I = Idx(0, b1, fermion=False)
                    i = 1 if b1 == b2 else 0
                    J = Idx(i, b2, fermion=False)
                    i = Idx(0, os)
                    a = Idx(0, vs)
                    e1 = Term(0.5,
                        [Sigma(I), Sigma(J), Sigma(i), Sigma(a)],
                        [Tensor([I, J, a, i], name, sym=sym)],
                        [BOperator(I, True), BOperator(J, True), FOperator(a, True), FOperator(i, False)],
                        [])
                    terms.append(e1)
    return Expression(terms)

def braE1(ospace, vspace):
    """
    Return left-projector onto a space of single excitations

    ospace (str): occupied space
    vspace (str): virtual space
    """
    i = Idx(0, ospace)
    a = Idx(0, vspace)
    operators = [FOperator(i,True), FOperator(a,False)]
    return Expression([Term(1.0, [], [Tensor([a,i],"")], operators, [])])

def braE2(o1, v1, o2, v2):
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
    operators = [FOperator(i,True), FOperator(j,True), FOperator(b,False), FOperator(a,False)]
    return Expression([Term(1.0, [], [Tensor([a,b,i,j],"")], operators, [])])

def braP1(space):
    """
    Return projection onto single Boson space
    """
    I = Idx(0, space, fermion=False)
    return Expression([Term(1.0, [], [Tensor([I],"")], [BOperator(I, False)], [])])

def braP2(space):
    """
    Return projection onto single Boson space
    """
    I = Idx(0, space, fermion=False)
    J = Idx(1, space, fermion=False)
    return Expression([Term(1.0, [], [Tensor([I,J],"")], [BOperator(I, False), BOperator(J, False)], [])])

def braP1E1(bspace, ospace, vspace):
    """
    Return left-projector onto a space of single excitations coupled to boson excitations

    vspace (str): boson space 
    ospace (str): occupied space
    vspace (str): virtual space
    """
    I = Idx(0, bspace, fermion=False)
    i = Idx(0, ospace)
    a = Idx(0, vspace)
    operators = [BOperator(I,False), FOperator(i,True), FOperator(a,False)]
    return Expression([Term(1.0, [], [Tensor([I,a,i],"")], operators, [])])

def braP2E1(b1space, b2space, ospace, vspace):
    """
    Return left-projector onto a space of single excitations coupled to boson excitations

    vspace (str): boson space 
    ospace (str): occupied space
    vspace (str): virtual space
    """
    I = Idx(0, b1space, fermion=False)
    i = 1 if b1space == b2space else 0
    J = Idx(i, b2space, fermion=False)
    i = Idx(0, ospace)
    a = Idx(0, vspace)
    operators = [BOperator(I,False), BOperator(J,False), FOperator(i,True), FOperator(a,False)]
    return Expression([Term(1.0, [], [Tensor([I,J,a,i],"")], operators, [])])

def braEip1(ospace):
    """
    Return left-projector onto a space of ionized determinants

    ospace (str): occupied space
    """
    i = Idx(0, ospace)
    operators = [FOperator(i,True)]
    return Expression([Term(1.0, [], [Tensor([i],"")], operators, [])])

def braEip2(o1, o2, v1):
    """
    Return left-projector onto a space of (trion) N-1 particle determinants

    o1 (str): first occupied space
    o2 (str): second occupied space
    v1 (str): virtual space
    """
    i = Idx(0, o1)
    a = Idx(0, v1)
    x = 1 if o2 == o1 else 0
    j = Idx(x, o2)
    operators = [FOperator(i,True), FOperator(j,True), FOperator(a,False)]
    return Expression([Term(1.0, [], [Tensor([a,i,j],"")], operators, [])])

def braEdip1(o1, o2):
    """
    Return left-projector onto a space of N-2 particle determinants

    o1 (str): first occupied space
    o2 (str): second occupied space
    """
    i = Idx(0, o1)
    x = 1 if o2 == o1 else 0
    j = Idx(x, o2)
    operators = [FOperator(i,True), FOperator(j,True)]
    return Expression([Term(1.0, [], [Tensor([i,j],"")], operators, [])])

def braEea1(space):
    """
    Return left-projector onto a space of N+1 electron states

    space (str): orbital space
    """
    a = Idx(0, space)
    operators = [FOperator(a,False)]
    return Expression([Term(1.0, [], [Tensor([a],"")], operators, [])])

def braEea2(o1, v1, v2):
    """
    Return left-projector onto a space of (trion) N+1 electron states

    o1 (str): occupied space
    v1 (str): first virtual space
    v2 (str): second virtual space
    """
    i = Idx(0, o1)
    a = Idx(0, v1)
    y = 1 if v2 == v1 else 0
    b = Idx(y, v2)
    operators = [FOperator(i, True), FOperator(b,False), FOperator(a, False)]
    return Expression([Term(1.0, [], [Tensor([a,b,i],"")], operators, [])])

def braEdea1(v1, v2):
    """
    Return left-projector onto a space of N+2 electron states

    v1 (str): first virtual space
    v2 (str): second virtual space
    """
    a = Idx(0, v1)
    y = 1 if v2 == v1 else 0
    b = Idx(y, v2)
    operators = [FOperator(b,False), FOperator(a,False)]
    return Expression([Term(1.0, [], [Tensor([a,b],"")], operators, [])])


def ketE1(ospace, vspace):
    """
    Return right-projector onto a space of single excitations

    ospace (str): occupied space
    vspace (str): virtual space
    """
    i = Idx(0, ospace)
    a = Idx(0, vspace)
    operators = [FOperator(a,True), FOperator(i,False)]
    return Expression([Term(1.0, [], [Tensor([i,a],"")], operators, [])])

def ketE2(o1, v1, o2, v2):
    """
    Return right-projector onto a space of double excitations

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
    operators = [FOperator(a,True), FOperator(b,True), FOperator(j,False), FOperator(i,False)]
    return Expression([Term(1.0, [], [Tensor([i,j,a,b],"")], operators, [])])

def ketEea1(space):
    """
    Return right-projector onto a space of N+1 electron states

    space (str): orbital space
    """
    a = Idx(0, space)
    operators = [FOperator(a,True)]
    return Expression([Term(1.0, [], [Tensor([a],"")], operators, [])])

def ketEea2(o1, v1, v2):
    """
    Return right-projector onto a space of trion N+1 electron states

    o1 (str): occupied space
    v1 (str): first virtual space
    v2 (str): second virtual space
    """
    i = Idx(0, o1)
    a = Idx(0, v1)
    y = 1 if v2 == v1 else 0
    b = Idx(y, v2)
    operators = [FOperator(a, True), FOperator(b,True), FOperator(i, False)]
    return Expression([Term(1.0, [], [Tensor([i,a,b],"")], operators, [])])

def ketEip1(space):
    """
    Return right-projector onto a space of N-1 electron states

    space (str): orbital space
    """
    i = Idx(0, space)
    operators = [FOperator(i,False)]
    return Expression([Term(1.0, [], [Tensor([i],"")], operators, [])])

def ketEip2(o1, o2, v1):
    """
    Return right-projector onto a space of trion N+1 electron states

    o1 (str): occupied space
    v1 (str): first virtual space
    o2 (str): second occupied space
    """
    i = Idx(0, o1)
    a = Idx(0, v1)
    x = 1 if o2 == o1 else 0
    j = Idx(x, o2)
    operators = [FOperator(a,True), FOperator(j, False), FOperator(i, False)]
    return Expression([Term(1.0, [], [Tensor([i,j,a],"")], operators, [])])

def ketEdea1(v1, v2):
    """
    Return right-projector onto a space of N+2 electron states

    v1 (str): first virtual space
    v2 (str): second virtual space
    """
    a = Idx(0, v1)
    y = 1 if v2 == v1 else 0
    b = Idx(y, v2)
    operators = [FOperator(a,True), FOperator(b,True)]
    return Expression([Term(1.0, [], [Tensor([a,b],"")], operators, [])])

def ketEdip1(o1, o2):
    """
    Return right-projector onto a space of N+2 electron states

    o1 (str): first occupied space
    o2 (str): second occupied space
    """
    i = Idx(0, o1)
    x = 1 if o2 == o1 else 0
    j = Idx(x, o2)
    operators = [FOperator(j,False), FOperator(i,False)]
    return Expression([Term(1.0, [], [Tensor([i,j],"")], operators, [])])

def commute(A, B):
    """ Return the commutator of two operators"""
    return A*B - B*A
