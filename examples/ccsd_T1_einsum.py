from wick.expression import *
from wick.ops import *
from wick.wick import apply_wick

H1 = one_e("f",["occ","vir"], norder=True)
H2 = two_e("I",["occ","vir"], norder=True)
H = H1 + H2

bra = braE1("occ", "vir")
T1 = E1("t", ["occ"], ["vir"])
T2 = E2("t", ["occ"], ["vir"])
T = T1 + T2

HT = commute(H,T)
HTT = commute(HT,T)
HTTT = commute(commute(commute(H2,T1),T1),T1)

S = bra*(H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT)
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final._print_einsum('T1'))
