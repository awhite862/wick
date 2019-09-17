from wick.expression import *
from wick.hamiltonian import *
from wick.wick import apply_wick

H1 = one_e("f",["occ","vir"], norder=True)

bra = projE1("occ", "vir")
T1 = E1("t", ["occ"], ["vir"])

HT = commute(H1,T1)
HTT = commute(HT,T1)
S = bra*(H1 + HT + 0.5*HTT)
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
