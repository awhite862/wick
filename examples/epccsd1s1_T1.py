from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, one_p, two_p, ep11
from wick.convenience import P1, E1, E2, EPS1, braE1, commute

H1 = one_e("f", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True, compress=True)
Hp = two_p("w") + one_p("G")
Hep = ep11("g", ["occ", "vir"], ["nm"], norder=True)
H = H1 + H2 + Hp + Hep

T1 = E1("t", ["occ"], ["vir"])
T2 = E2("t", ["occ"], ["vir"])
S1 = P1("s", ["nm"])
U11 = EPS1("u", ["nm"], ["occ"], ["vir"])
T = T1 + T2 + S1 + U11
bra = braE1("occ", "vir")
HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(commute(commute(H2, T1), T1), T1)
S = bra*(H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT)
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
