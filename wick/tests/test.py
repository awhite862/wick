from wick.index import Idx
from wick.expression import *
from wick.wick import apply_wick

s = 0.5
sums = []
tensors = []
operators = []
deltas = []

indices = [Idx(0,"vir"),Idx(1,"vir")]
indices2 = [Idx(0,"occ"),Idx(1,"occ")]
sums.append(Sigma(indices[0]))
sums.append(Sigma(indices[1]))
sums.append(Sigma(Idx(0,"occ")))
tensors.append(Tensor(indices2, "f"))
tensors.append(Tensor(indices, "f"))
operators.append(FOperator(indices[0], False))
operators.append(FOperator(indices[1], True))
operators.append(FOperator(indices2[0], True))
operators.append(FOperator(indices2[1], False))

t = Term(s,sums, tensors, operators, deltas)

e = Expression([t])
print(e)
print(e._print_str())
x = apply_wick(e)
print(x)
x.resolve()
y = AExpression(Ex=x)
y.simplify()
print(y)
print(y._print_str())
