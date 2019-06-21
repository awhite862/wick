from wick.index import Idx
from wick.expression import *
from wick.wick import apply_wick

s = 0.5
sums = []
tensors = []
operators = []
deltas = []

indices = [Idx("a","vir"),Idx("b","vir")]
indices2 = [Idx("i","occ"),Idx("j","occ")]
sums.append(Sigma(indices[0]))
sums.append(Sigma(indices[1]))
sums.append(Sigma(Idx("i","occ")))
tensors.append(Tensor(indices2, "f"))
tensors.append(Tensor(indices, "f"))
operators.append(Operator(indices[0], False))
operators.append(Operator(indices[1], True))
operators.append(Operator(indices2[0], True))
operators.append(Operator(indices2[1], False))

t = Term(s,sums, tensors, operators, deltas)

e = Expression([t])
print(e)
x = apply_wick(e)
print(x)
x.resolve()
print(x)
