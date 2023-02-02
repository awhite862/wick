from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, E1, E2, braE1, commute

PYTHON_FILE_TAB = "    "


def space_idx_formatter(name, space_list):
    s = f"{name}["

    for i, space in enumerate(space_list):
        s += f"{space}_idx"
        if i != len(space_list) - 1:
            s += ", "
    s += "]"

    return s

def einsum_str_formatter(sstr, fstr, istr, tstr):
    einsum_str = "\'" + istr + "->" + fstr + "\'"
    return f"{float(sstr): 12.6f} * einsum({einsum_str:20s}{tstr})"

def gen_einsum_fxn(final, name_str="get", return_str="tmp", arg_str_list=None, file_obj=None):
    if arg_str_list is None:
        arg_str_list = ["t1ov", "t2oovv", "h1e", "h2e", "occ_idx", "vir_idx"]

    arg_str = ""
    for iarg, arg in enumerate(arg_str_list):
        arg_str += f"{arg}, " if iarg != len(arg_str_list) - 1 else f"{arg}"

    function_str = '''def %s(%s):\n''' % (name_str, arg_str)
    function_str += PYTHON_FILE_TAB + "tmp = 0.0\n"
    function_str += PYTHON_FILE_TAB + "t1  = t1ov.transpose()\n"
    function_str += PYTHON_FILE_TAB + "t2  = t2oovv.transpose(2, 3, 0, 1)\n\n"

    einsum_str = final._print_einsum(return_str, exprs_with_space=[H],
                                     space_idx_formatter=space_idx_formatter, einsum_str_formatter=einsum_str_formatter)
    einsum_str = einsum_str.split("\n")

    for i, line in enumerate(einsum_str):
        function_str += f"{PYTHON_FILE_TAB}{line}\n"

    function_str += f"{PYTHON_FILE_TAB}\n"
    function_str += f"{PYTHON_FILE_TAB}return tmp\n"

    file_obj.write(function_str)


H1 = one_e("h1e", ["occ", "vir"], norder=True)
H2 = two_e("h2e", ["occ", "vir"], norder=True)
H = H1 + H2

bra = braE1("occ", "vir")
T1 = E1("t1", ["occ"], ["vir"])
T2 = E2("t2", ["occ"], ["vir"])
T = T1 + T2

HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(commute(commute(H2, T1), T1), T1)

S = bra*(H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT)
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)

with open("ccsd_T1_einsum_idx.out", "w") as f:
    gen_einsum_fxn(final, name_str="get_ccsd_t1", file_obj=f)
