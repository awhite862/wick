from wick.index import Idx
from wick.expression import *
from wick.hamiltonian import one_e, two_e, E1, get_sym, commute
from wick.wick import apply_wick

i = Idx(0,"occ")
a = Idx(0,"vir")
j = Idx(1,"occ")
b = Idx(1,"vir")

T1 = E1("t", ["occ"], ["vir"])
sym = get_sym(True)
T2 = Expression([Term(0.25,
    [Sigma(i), Sigma(a), Sigma(j), Sigma(b)],
    [Tensor([a, b, i, j], "t",sym=sym)],
    [FOperator(a, True), FOperator(i, False), FOperator(b, True), FOperator(j, False)],
    [])])
T = T1 + T2

L1 = Expression([Term(1.0,
    [Sigma(i), Sigma(a)],
    [Tensor([i, a], "L")],
    [FOperator(i, True), FOperator(a, False)],
    [])])
sym = get_sym(True)
L2 = Expression([Term(0.25,
    [Sigma(i), Sigma(a), Sigma(j), Sigma(b)],
    [Tensor([i, j, a, b], "L",sym=sym)],
    [FOperator(i, True), FOperator(a, False), FOperator(j, True), FOperator(b, False)],
    [])])
L = L1 + L2

# ov block
operators = [FOperator(a,True), FOperator(i,False)]
pvo = Expression([Term(1.0, [], [Tensor([i,a],"")], operators, [])])

PT = commute(pvo, T)
PTT = commute(PT, T)
mid = pvo + PT + 0.5*PTT
full = mid + L*mid
full = L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.simplify()
final.sort()
print("P_{ov} = ")
print(final._print_str())

# vv block
operators = [FOperator(a,True), FOperator(b,False)]
pvv = Expression([Term(1.0, [], [Tensor([b,a],"")], operators, [])])

PT = commute(pvv, T)
PTT = commute(PT, T)
mid = pvv + PT + 0.5*PTT
full = mid + L*mid
full = L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.simplify()
final.sort()
final.sort_tensors()
print("P_{vv} = ")
print(final._print_str())

# oo block
operators = [FOperator(j,False), FOperator(i,True)]
poo = Expression([Term(-1.0, [], [Tensor([j,i],"")], operators, [])])

PT = commute(poo, T)
PTT = commute(PT, T)
mid = poo + PT + 0.5*PTT
full = mid + L*mid
full = L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.simplify()
final.sort()
final.sort_tensors()
print("P_{oo} = ")
print(final._print_str())

# vo block
operators = [FOperator(i,True), FOperator(a,False)]
pvo = Expression([Term(1.0, [], [Tensor([a,i],"")], operators, [])])

PT = commute(pvo, T)
PTT = commute(PT, T)
mid = pvo + PT + 0.5*PTT
full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.simplify()
final.sort()
final.sort_tensors()
print("P_{vo} = ")
print(final._print_str())
