from wick.index import Idx
from wick.expression import *
from wick.hamiltonian import one_e, two_e
from wick.wick import apply_wick

H1 = one_e("f",["occ","vir"])
H2 = two_e("I",["occ","vir"])

H = H1# + H2
i = Idx(0,"occ")
a = Idx(0,"vir")
operators = [Operator(i,True), Operator(a,False)]
bra = Expression([Term(1.0, [], [], operators, [])])
ket = Expression([Term(1.0,
    [Sigma(i), Sigma(a)],
    [Tensor([a, i], "c")],
    [Operator(a, True), Operator(i, False)],
    [])])

HC = H*ket
S = bra*HC
temp = Expression(HC.terms[0:1])
tt = bra*temp
out = apply_wick(tt)
out.resolve()
print(out._print_str())
