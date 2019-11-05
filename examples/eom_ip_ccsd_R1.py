from wick.expression import *
from wick.hamiltonian import *
from wick.wick import apply_wick

H1 = one_e("f",["occ","vir"], norder=True)
H2 = two_e("I",["occ","vir"], norder=True)
H = H1 + H2

bra = braEip1("occ")
T1 = E1("t", ["occ"], ["vir"])
T2 = E2("t", ["occ"], ["vir"])

R1 = Eip1("r", ["occ"])
R2 = Eip2("r", ["occ"], ["vir"])

T = T1 + T2
R = R1 + R2
HT = commute(H,T)
HTT = commute(HT,T)
HTTT = commute(HTT,T)
HTTTT = commute(HTTT,T)

S0 = (H + HT + (1.0/2.0)*HTT)
E0 = apply_wick(S0)
E0.resolve()

S = bra*(H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT + (1/24.0)*HTTTT - E0)*R

out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
