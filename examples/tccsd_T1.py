from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, E1, E2, braE1, commute


index_key = {
    "occ": "ijklmno",
    "oa": "IJKLMNO",
    "va": "ABCDEFG",
    "vir": "abcdefg"}

H1 = one_e("f", ["occ", "oa", "va", "vir"], norder=True, index_key=index_key)
H2 = two_e("I", ["occ", "oa", "va", "vir"], norder=True, index_key=index_key)
H = H1 + H2

bra = braE1("occ", "vir", index_key=index_key)
T1 = E1("t", ["occ", "oa"], ["va", "vir"], index_key=index_key)
T2 = E2("t", ["occ", "oa"], ["va", "vir"], index_key=index_key)
T = T1 + T2

HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(commute(commute(H2, T1), T1), T1)

S = bra*H
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print("Constant:")
print(final)
print("")

S = bra*HT
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print("Linear:")
print(final)
print("")

S = Fraction(1, 2)*bra*HT
out = apply_wick(S)
out.resolve()
print("Quadratic:")
print(final)
print("")

S = Fraction(1, 6)*bra*HTTT
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print("Cubic:")
print(final)
print("")
