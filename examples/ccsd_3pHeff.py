from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, E1, E2, commute
from wick.convenience import ketEea2, braEea2, ketEip2, braEip2

H1 = one_e("f", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True)
H = H1 + H2

T1 = E1("t", ["occ"], ["vir"])
T2 = E2("t", ["occ"], ["vir"])
T = T1 + T2

HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(HTT, T)
HTTTT = commute(HTTT, T)
Hbar = H + HT + Fraction('1/2')*HTT
Hbar += Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT

# vovvvo piece
ket = ketEea2("occ", "vir", "vir")
bra = braEea2("occ", "vir", "vir")
S = bra*Hbar*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
final.transpose((0, 3, 1, 4, 5, 2))
print("W_{vovvvo} = ")
print(final)

# oovovo piece
ket = ketEip2("occ", "occ", "vir")
bra = braEip2("occ", "occ", "vir")
S = -1*bra*Hbar*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
final.transpose((3, 4, 0, 1, 5, 2))
print("W_{oovovo} = ")
print(final)
