from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_p, braP1


Hp = one_p("G")
bra = braP1("nm")
S = bra*Hp
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final)
