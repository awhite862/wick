from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, E1, braE1

H1 = one_e("f", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True)

H = H1 + H2
bra = braE1("occ", "vir")
ket = E1("c", ["occ"], ["vir"])

HC = H*ket
S = bra*HC
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
