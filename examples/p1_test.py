from wick.index import Idx
from wick.expression import Expression, Term, AExpression
from wick.operator import BOperator, Tensor
from wick.hamiltonian import one_p
from wick.wick import apply_wick


Hp = one_p("G")
I = Idx(0,"nm",fermion=False)
bra = Expression([Term(1.0, [], [Tensor([],"")], [BOperator(I, True)], [])])
S = bra*Hp
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
final.simplify()
final.sort()
print(final._print_str())
