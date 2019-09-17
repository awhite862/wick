from wick.expression import *
from wick.hamiltonian import *
from wick.wick import apply_wick

H1 = one_e("f",["occ","vir"], norder=True)
H2 = two_e("I",["occ","vir"], norder=True)

H = H1 + H2
bra = projE1("occ", "vir")
ket = E1("c", ["occ"], ["vir"])

HC = H*ket
S = bra*HC
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final._print_str())
