from wick.expression import AExpression
from wick.convenience import *
from wick.wick import apply_wick


H = two_p("w")
bra = braP2("nm")
S2 = P2("S2old", ["nm"])
HT = commute(H,S2)
HTT = commute(HT,S2)
S = bra*(H + HT + 0.5*HTT)
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
