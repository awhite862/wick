from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, E1, E2, ketE1, commute

H1 = one_e("f", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True)
H = H1 + H2

T1 = E1("t", ["occ"], ["vir"])
T2 = E2("t", ["occ"], ["vir"])
T = T1 + T2

L1 = E1("L", ["vir"], ["occ"])
L2 = E2("L", ["vir"], ["occ"])
L = L1 + L2

ket = ketE1("occ", "vir")

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
S1 = L*S
out1 = apply_wick(S1)
out1.resolve()
ex1 = AExpression(Ex=out1)
ex1 = ex1.get_connected()
ex1.sort_tensors()

# Subtract those terms that sum to zero
S2 = L*ket*Hbar
out2 = apply_wick(S2)
out2.resolve()
ex2 = AExpression(Ex=out2)
ex2 = ex2.get_connected()
ex2.sort_tensors()
print(ex1 - ex2)
