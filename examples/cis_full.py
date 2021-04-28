from wick.expression import AExpression
from wick.convenience import *
from wick.wick import apply_wick

H1 = one_e("f", ["occ", "vir"])
H2 = two_e("I", ["occ", "vir"])

H = H1 + H2
bra = braE1("occ", "vir")
ket = E1("c", ["occ"], ["vir"])

HC = H*ket
S = bra*HC
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
