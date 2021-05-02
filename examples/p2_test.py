from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import two_p, P2, braP2, commute


H = two_p("w")
bra = braP2("nm")
S2 = P2("S2old", ["nm"])
HT = commute(H, S2)
HTT = commute(HT, S2)
S = bra*(H + HT + 0.5*HTT)
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
