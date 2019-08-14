from wick.index import Idx
from wick.expression import *
from wick.hamiltonian import one_e, two_e, get_sym
from wick.wick import apply_wick

H1 = one_e("f",["occ","vir"], norder=True)
H2 = two_e("I",["occ","vir"], norder=True)

H = H1 + H2
i = Idx(0,"occ")
j = Idx(1,"occ")
a = Idx(0,"vir")
b = Idx(1,"vir")
operators = [FOperator(i,True), FOperator(a,False), FOperator(j,True), FOperator(b,False)]

sym = get_sym(True)
bra = Expression([Term(1.0, [], [Tensor([i, j, a, b], "")], operators, [])])
ket = Expression(
    [Term(0.25,
        [Sigma(i), Sigma(a), Sigma(j), Sigma(b)],
        [Tensor([a, b, i, j], "c", sym=sym)],
        [FOperator(a, True), FOperator(i, False), FOperator(b, True), FOperator(j, False)],
        []),
    Term(1.0,[], [Tensor([], "c")], [], [])
    ])

HC = H*ket
S = bra*HC
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.simplify()
final.sort()
print("Sigma2")
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

