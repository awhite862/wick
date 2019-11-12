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

# ov piece
ket = ketE1("occ", "vir")
S = (H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT)*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
print("F_{ov} = ")
print(final)

# vv piece
ket = ketEea1("vir")
bra = braEea1("vir")
S = bra*(H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT)*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
print("F_{vv} = ")
print(final)

# oo piece
ket = ketEip1("occ")
bra = braEip1("occ")
S = -1.0*bra*(H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT)*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
final.transpose((1,0))
print("F_{oo} = ")
print(final)
#print(final._print_einsum())

# vvoo piece
ket = ketE2("occ", "vir", "occ", "vir")
S = (H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT + (1/24.0)*HTTTT)*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
print("W_{oovv} = ")
print(final)

# vovv piece
ket = ketEea2("occ", "vir", "vir")
bra = braEea1("vir")
S = bra*(H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT + (1/24.0)*HTTTT)*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
print("W_{vovv} = ")
print(final)

# ooov piece
ket = ketEip2("occ", "occ", "vir")
bra = braEip1("occ")
S = -1.0*bra*(H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT + (1/24.0)*HTTTT)*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
final.transpose((1,2,0,3))
print("W_{ooov} = ")
print(final)
#print(final._print_einsum())

# vvvv piece
ket = ketEdea1("vir", "vir")
bra = braEdea1("vir", "vir")
S = bra*(H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT + (1/24.0)*HTTTT)*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
print("W_{vvvv} = ")
print(final)
#print(final._print_einsum())

# oooo piece
ket = ketEdip1("occ", "occ")
bra = braEdip1("occ", "occ")
S = bra*(H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT + (1/24.0)*HTTTT)*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
final.transpose((2,3,0,1))
print("W_{oooo} = ")
print(final)
#print(final._print_einsum())

# voov piece
ket = ketE1("occ", "vir")
bra = braE1("occ", "vir")
S = bra*(H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT + (1/24.0)*HTTTT)*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
final.transpose((0,2,1,3))
print("W_{voov} = ")
print(final)
#print(final._print_einsum())

# vvvo piece
ket = ketEea1("vir")
bra = braEea2("occ", "vir", "vir")
S = bra*(H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT + (1/24.0)*HTTTT)*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
final.transpose((0,1,3,2))
print("W_{vvvo} = ")
print(final)
#print(final._print_einsum())

# ovoo piece
ket = ketEip1("occ")
bra = braEip2("occ", "occ", "vir")
S = -1.0*bra*(H + HT + (1.0/2.0)*HTT + (1/6.0)*HTTT + (1/24.0)*HTTTT)*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
final.transpose((3,0,1,2))
print("W_{ovoo} = ")
print(final)
#print(final._print_einsum())
