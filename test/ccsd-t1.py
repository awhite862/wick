from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, E1, E2, braE1, commute

def space_formatter(space_list):
    print("Space list:", space_list)
    return "haha"

H1 = one_e("h1e", ["occ", "vir"], norder = True)
H2 = two_e("h2e", ["occ", "vir"], norder = True)
H = H1 + H2

bra = braE1("occ", "vir")
T1 = E1("t1", ["occ"], ["vir"])
T2 = E2("t2", ["occ"], ["vir"])
T = T1 + T2

HT   = commute(H, T)
HTT  = commute(HT, T)
HTTT = commute(commute(commute(H2, T1), T1), T1)

S = bra*(H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT)
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)

with open("ccsd-t1.out", "w") as f:
    f.write("\n\n# T1 part\n")
    f.write(final._print_einsum('T1', exprs_with_space = [H], space_formatter = space_formatter))

    f.write("\n\n# T2 part\n")
    f.write(final._print_einsum('T2', exprs_with_space = [H], space_formatter = space_formatter))