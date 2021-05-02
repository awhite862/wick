from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, braE1, ketE1, braE2, ketE2

H1 = one_e("f", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True)
H = H1 + H2

# first derivative wrt X*
bra = braE1("occ", "vir")
S = bra*H
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("dE/dX* =")
print(final)

# first derivative wrt X
ket = ketE1("occ", "vir")
S = H*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final.transpose((1, 0))
print("dE/dX =")
print(final)

print("")
# second derivative wrt X*X*
bra = braE2("occ", "vir", "occ", "vir")
S = bra*H
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("d^2E/dX*dX* =")
print(final)

# second derivative wrt X*X
ket = ketE1("occ", "vir")
bra = braE1("occ", "vir")
S = bra*H*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final.transpose((0, 1, 3, 2))
print("d^2E/dX*dX =")
print(final)

# second derivative wrt XX
ket = ketE2("occ", "vir", "occ", "vir")
S = H*ket
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
final.transpose((2, 3, 0, 1))
print("d^2E/dXdX =")
print(final)
