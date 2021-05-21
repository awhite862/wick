from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, E1, E2, PE1, ketE2, commute

H1 = one_e("f", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True)
H = H1 + H2

T1 = E1("t", ["occ"], ["vir"])
T2 = E2("t", ["occ"], ["vir"])
T = T1 + T2

L1 = E1("L", ["vir"], ["occ"])
L2 = E2("L", ["vir"], ["occ"])
L = L1 + L2

ket = ketE2("occ", "vir", "occ", "vir")

HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(HTT, T)
HTTTT = commute(HTTT, T)
Hbar = H + HT + Fraction('1/2')*HTT
Hbar += Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT

# Pieces not proportaional to lambda
S = Hbar*ket
out = apply_wick(S)
out.resolve()
ex = AExpression(Ex=out)
ex = ex.get_connected()
ex.sort_tensors()
print(ex)
print("")

# Connected pieces proportional to Lambda
S = L*S
out = apply_wick(S)
out.resolve()
ex = AExpression(Ex=out)
ex = ex.get_connected()
ex.sort_tensors()
print(ex)
print("")

# Disonnected pieces proportional to Lambda
P1 = PE1("occ", "vir")
S = (H + HT)*P1*L*ket
out = apply_wick(S)
out.resolve()
ex = AExpression(Ex=out)
ex.sort_tensors()
print(ex)
