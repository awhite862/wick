from wick.expression import *
from wick.hamiltonian import *
from wick.wick import apply_wick

H1 = one_e("f",["occ","vir"], norder=True)
H2 = two_e("I",["occ","vir"], norder=True)
H = H1 + H2

T1 = E1("t", ["occ"], ["vir"])
T2 = E2("t", ["occ"], ["vir"])
T = T1 + T2

HT = commute(H,T)
HTT = commute(HT,T)
HTTT = commute(HTT,T)
HTTTT = commute(HTTT,T)

# CCSD energy 
S0 = (H + HT + (1.0/2.0)*HTT)
E0 = apply_wick(S0)
E0.resolve()

# ia piece
ket = ketE1("occ", "vir")
S = (H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT)*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("F_{ia} = ")
print(final)

# ba piece
ket = ketF1("vir")
bra = projF1("vir")
S = bra*(H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT - E0)*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("F_{ba} = ")
print(final)

# ij piece
ket = ketF_1("occ")
bra = projF_1("occ")
S = bra*(H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT - E0)*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("F_{ji} = ")
print(final)

