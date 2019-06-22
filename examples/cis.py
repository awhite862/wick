from wick.expression import *
from wick.hamiltonian import one_e, two_e
from wick.wick import apply_wick

H1 = one_e("f",["occ","vir"])
H2 = two_e("I",["occ","vir"])

H = H1 + H2
