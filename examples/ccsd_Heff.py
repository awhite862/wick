from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, E1, E2, commute
from wick.convenience import ketE1, ketE2, ketEip1, ketEea1
from wick.convenience import ketEea2, ketEip2, ketEdea1, ketEdip1
from wick.convenience import braE1, braEip1, braEea1
from wick.convenience import braEea2, braEip2, braEdea1, braEdip1

H1 = one_e("f", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True, compress=True)
H = H1 + H2

T1 = E1("t", ["occ"], ["vir"])
T2 = E2("t", ["occ"], ["vir"])
T = T1 + T2

HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(HTT, T)
HTTTT = commute(HTTT, T)

Hbar = H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT

# ov piece
ket = ketE1("occ", "vir")
S = Hbar*ket
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
S = bra*Hbar*ket
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
S = -1*bra*Hbar*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
final.transpose((1, 0))
print("F_{oo} = ")
print(final)

Hbar += Fraction('1/24')*HTTTT

# vvoo piece
ket = ketE2("occ", "vir", "occ", "vir")
S = Hbar*ket
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
S = bra*Hbar*ket
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
S = -1*bra*Hbar*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
final.transpose((1, 2, 0, 3))
print("W_{ooov} = ")
print(final)

# vvvv piece
ket = ketEdea1("vir", "vir")
bra = braEdea1("vir", "vir")
S = bra*Hbar*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
print("W_{vvvv} = ")
print(final)

# oooo piece
ket = ketEdip1("occ", "occ")
bra = braEdip1("occ", "occ")
S = bra*Hbar*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
final.transpose((2, 3, 0, 1))
print("W_{oooo} = ")
print(final)

# voov piece
ket = ketE1("occ", "vir")
bra = braE1("occ", "vir")
S = bra*Hbar*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
final.transpose((0, 2, 1, 3))
print("W_{voov} = ")
print(final)

# vvvo piece
ket = ketEea1("vir")
bra = braEea2("occ", "vir", "vir")
S = bra*Hbar*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
final.transpose((0, 1, 3, 2))
print("W_{vvvo} = ")
print(final)

# ovoo piece
ket = ketEip1("occ")
bra = braEip2("occ", "occ", "vir")
S = -1*bra*Hbar*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final = final.get_connected()
final.transpose((3, 0, 1, 2))
print("W_{ovoo} = ")
print(final)
