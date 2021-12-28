# Copyright (c) 2020-2021 Alec White
# Licensed under the MIT License (see LICENSE for details)
from itertools import product
from .operator import BOperator, FOperator, Projector, Delta
from .expression import Term, Expression
from .index import is_occupied


def valid_contraction(o1, o2, occ=None):
    if o1.idx.space != o2.idx.space:
        return False

    if isinstance(o1, FOperator) and isinstance(o2, FOperator):
        if is_occupied(o1.idx, occ=occ) and o1.ca and not o2.ca:
            return True
        if not is_occupied(o1.idx, occ=occ) and not o1.ca and o2.ca:
            return True
        return False
    elif isinstance(o1, BOperator) and isinstance(o2, BOperator):
        if (not o1.ca and o2.ca):
            return True
        return False
    elif type(o1) is not type(o1):
        return False
    else:
        return True


def pair_list(lst, occ=None):
    n = len(lst)
    assert n % 2 == 0
    if n < 2:
        return []
    elif n == 2:
        if valid_contraction(lst[0], lst[1], occ=occ):
            return [[(lst[0], lst[1])]]
        else:
            return []
    else:
        ltmp = lst[1:]
        yy = lst[0]
        plist = []
        for i, x in enumerate(ltmp):
            if valid_contraction(yy, x):
                p1 = [(lst[0], x)]
                remainder = pair_list(ltmp[:i] + ltmp[i + 1:])
                plist += [r + p1 for r in remainder]
        return plist


def find_pair(i, ipairs):
    for p in ipairs:
        if i in p:
            return p
    return None


def get_sign(ipairs):
    ncross = 0
    for p in ipairs:
        i, j = p
        for x1 in range(i + 1, j):
            p1 = find_pair(x1, ipairs)
            if p1 is None:
                continue
            x2 = p1[0] if p1[1] == x1 else p1[1]
            if x2 > j or x2 < i:
                ncross += 1

    assert ncross % 2 == 0
    ncross = ncross // 2
    return 1 if ncross % 2 == 0 else -1


def split_operators(ops):
    ps = []
    for i, op in enumerate(ops):
        if isinstance(op, Projector):
            ps.append(i)

    if len(ps) == 0:
        return [ops]
    starts = [0] + [x + 1 for x in ps]
    ends = ps + [len(ops)]
    olists = []
    for s, e in zip(starts, ends):
        olists.append(ops[s:e])
    return olists


def apply_wick(e, occ=None):
    to = []
    # loop over terms
    for temp in e.terms:
        olists = split_operators(temp.operators)
        if not any(olists):
            to.append(temp.copy())
            continue

        dos = []
        sos = []
        # if member of the product has an odd number of operators,
        # then we are done
        oparity = [len(operators) % 2 == 0 for operators in olists]
        if not all(oparity):
            continue
        for operators in olists:
            if len(operators) == 0:
                continue
            # loop to find a contraction
            plist = pair_list(operators)
            ds = []
            ss = []
            for pairs in plist:
                good = bool(pairs)
                ipairs = []
                deltas = []
                for p in pairs:
                    oi, oj = p
                    if oi.idx.space != oj.idx.space:
                        good = False
                        break
                    if not oi.idx.fermion:
                        i1 = oi.idx
                        i2 = oj.idx
                        deltas.append(Delta(i1, i2))
                    elif (is_occupied(oi.idx, occ=occ) and oi.ca and not oj.ca) or (
                            not is_occupied(oi.idx, occ=occ) and not oi.ca and oj.ca):
                        i = operators.index(oi)
                        j = operators.index(oj)
                        ipairs.append((i, j))
                        i1 = oi.idx
                        i2 = oj.idx
                        deltas.append(Delta(i1, i2))
                    else:
                        good = False
                        break

                # append to output
                if good:
                    ds.append(deltas)
                    ss.append(get_sign(ipairs))

            dos.append(ds)
            sos.append(ss)

        # If there are no contractions, continue
        if not sos:
            assert len(dos) == 0
            continue
        for di, si in zip(product(*dos), product(*sos)):
            assert si
            assert di
            sign = 1
            for s in si:
                sign *= s
            deltas = []
            for d in di:
                deltas += d
            t1 = Term(
                sign*temp.scalar,
                [s.copy() for s in temp.sums],
                [t.copy() for t in temp.tensors],
                [], deltas + [d.copy() for d in temp.deltas],
                index_key=temp.index_key)
            to.append(t1)

    o = Expression(to)
    if o.are_operators():
        raise Exception("Application of Wick's theorem has failed!")

    return o
