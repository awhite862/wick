from wick.operator import Tensor
from wick.expression import AExpression, Term, Expression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, E0, E1, E2, braE1, braE2

H1 = one_e("f", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True)

H = H1 + H2

C0 = E0("c")
C1 = E1("c", ["occ"], ["vir"])
C2 = E2("c", ["occ"], ["vir"])

ket = C0 + C1 + C2
HC = H*ket

bra = braE2("occ", "vir", "occ", "vir")
S = bra*HC
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print("Sigma2")
print(final)

bra = braE1("occ", "vir")
S = bra*HC
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print("Sigma1")
print(final)

bra = Expression([Term(1, [], [Tensor([], "")], [], [])])
S = bra*HC
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print("Sigma0")
print(final)
