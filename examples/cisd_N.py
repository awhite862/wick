from wick.index import Idx
from wick.expression import *
from wick.hamiltonian import one_e, two_e, E1, E2
from wick.wick import apply_wick

H1 = one_e("f",["occ","vir"], norder=True)
H2 = two_e("I",["occ","vir"], norder=True)

H = H1 + H2
i = Idx(0,"occ")
j = Idx(1,"occ")
a = Idx(0,"vir")
b = Idx(1,"vir")

C0 = Expression([Term(1.0,[], [Tensor([], "c")], [], [])])
C1 = E1("c", ["occ"], ["vir"])
C2 = E2("c", ["occ"], ["vir"])

ket = C0 + C1 + C2
HC = H*ket

operators = [FOperator(i,True), FOperator(a,False), FOperator(j,True), FOperator(b,False)]
bra = Expression([Term(1.0, [], [Tensor([i, j, a, b], "")], operators, [])])
S = bra*HC
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.simplify()
final.sort()
print("Sigma2")
print(final._print_str())

operators = [FOperator(i,True), FOperator(a,False)]
bra = Expression([Term(1.0, [], [Tensor([i, a], "")], operators, [])])
S = bra*HC
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.simplify()
final.sort()
print("Sigma1")
print(final._print_str())


bra = Expression([Term(1.0, [], [Tensor([], "")], [], [])])
S = bra*HC
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.simplify()
final.sort()
print("Sigma0")
print(final._print_str())

