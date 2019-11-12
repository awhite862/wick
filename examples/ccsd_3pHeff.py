from wick.expression import *
from wick.ops import *
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

# vovvvo piece
ket = ketEea2("occ", "vir", "vir")
bra = braEea2("occ", "vir", "vir")
S = bra*(H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT + (1/24.0)*HTTTT)*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
final.transpose((0,3,1,4,5,2))
print("W_{vovvvo} = ")
print(final)
#print(final._print_einsum())

# oovovo piece
ket = ketEip2("occ", "occ", "vir")
bra = braEip2("occ", "occ", "vir")
S = -1.0*bra*(H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT + (1/24.0)*HTTTT)*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
final.transpose((3,4,0,1,5,2))
print("W_{oovovo} = ")
print(final)
#print(final._print_einsum())
