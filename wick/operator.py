# Copyright (c) 2020-2021 Alec White
# Licensed under the MIT License (see LICENSE for details)
from .index import Idx
from .index import idx_copy
from .index import is_occupied


class Projector(object):
    """
    Projector onto the vacuum
    """
    def __init__(self):
        self.idx = None

    def __eq__(self, other):
        return isinstance(other, Projector)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "P"

    def _inc(self, i):
        return self

    def _print_str(self, imap):
        return "P"

    def copy(self):
        return self

    def dagger(self):
        return self


class FOperator(object):
    """
    Fermion creation/annihilation operators

    Attributes:
        idx (Idx): Index of operator
        ca (Bool): Creation operator?
    """
    def __init__(self, idx, ca):
        self.idx = idx
        assert self.idx.fermion
        self.ca = ca

    def __eq__(self, other):
        if isinstance(other, FOperator):
            return self.idx == other.idx\
                and self.ca == other.ca
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        if self.ca:
            return "a^{\\dagger}_" + str(self.idx)
        else:
            return "a_" + str(self.idx)

    def _inc(self, i):
        """Increment indices"""
        return FOperator(Idx(self.idx.index + i, self.idx.space), self.ca)

    def _print_str(self, imap):
        if self.ca:
            return "a^{\\dagger}_" + imap[self.idx]
        else:
            return "a_" + imap[self.idx]

    def qp_creation(self, occ=None):
        if (not is_occupied(self.idx, occ=occ)) and self.ca:
            return True
        elif (is_occupied(self.idx, occ=occ)) and not self.ca:
            return True
        else:
            return False

    def qp_annihilation(self, occ=None):
        return not self.qp_creation(occ=occ)

    def copy(self):
        return FOperator(idx_copy(self.idx), self.ca)

    def dagger(self):
        return FOperator(idx_copy(self.idx), not self.ca)


class BOperator(object):
    """
    Boson creation/annihilation operators

    Attributes:
        idx (Idx): Index of operator
        ca (Bool): Creation operator?
    """
    def __init__(self, idx, ca):
        self.idx = idx
        assert not self.idx.fermion
        self.ca = ca

    def __eq__(self, other):
        if isinstance(other, BOperator):
            return self.idx == other.idx\
                and self.ca == other.ca
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        if self.ca:
            return "b^{\\dagger}_" + str(self.idx)
        else:
            return "b_" + str(self.idx)

    def _inc(self, i):
        """Increment indices"""
        return BOperator(
            Idx(self.idx.index + i, self.idx.space, fermion=False), self.ca)

    def _print_str(self, imap):
        if self.ca:
            return "b^{\\dagger}_" + imap[self.idx]
        else:
            return "b_" + imap[self.idx]

    def qp_creation(self):
        return self.ca

    def qp_annihilation(self):
        return not self.qp_creation()

    def copy(self):
        return BOperator(idx_copy(self.idx), self.ca)

    def dagger(self):
        return BOperator(idx_copy(self.idx), not self.ca)


class TensorSym(object):
    """
    Representation of tensor permutational symmetry

    Attributes:
        plist (lsit): List of tuples representing permutations of indices
        signs (list): List of signs representing the signes of each permutation
        tlist (list): List of permutation, sign pairs
    """
    def __init__(self, plist, signs):
        self.plist = plist
        self.signs = signs
        self.tlist = list(zip(plist, signs))


class Tensor(object):
    """
    Tensor class

    Attributes:
        indices (list): List of indices
        name (str): Name of the tensor
        sym (TensorSym): Permutational symmetry of tensor
    """
    def __init__(self, indices, name, sym=None):
        self.indices = indices
        self.name = name
        if sym is None:
            self.sym = TensorSym(
                [tuple(range(len(indices)))], [1])
        else:
            self.sym = sym

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return self.indices == other.indices \
                and self.name == other.name
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if len(self.indices) < len(other.indices):
            return True
        elif len(self.indices) == len(other.indices):
            if self.name < other.name:
                return True
            elif self.name == other.name:
                return self.indices < other.indices
            else:
                return False
        else:
            return False

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __hash__(self):
        ss = str(self.name)
        for i in self.indices:
            ss += str(i)
        return hash(ss)

    def __repr__(self):
        istr = str()
        for idx in self.indices:
            istr += str(idx.index)
        return self.name + "_{" + istr + "}"

    def _inc(self, i):
        indices = [Idx(ii.index + i, ii.space) for ii in self.indices]
        return Tensor(indices, self.name, sym=self.sym)

    def ilist(self):
        ilist = []
        for idx in self.indices:
            if idx not in ilist:
                ilist.append(idx)
        return ilist

    def _istr(self, imap):
        out = str()
        for idx in self.indices:
            out += imap[idx]
        return out

    def _print_str(self, imap):
        temp = self.name
        if len(temp) == 0:
            return str()
        istr = str()
        for idx in self.indices:
            istr += imap[idx]
        return self.name + "_{" + istr + "}"

    def transpose(self, perm):
        assert len(self.indices) == len(perm)
        newindices = []
        for p in perm:
            newindices.append(self.indices[p])
        self.indices = newindices

    def copy(self):
        newindices = [idx_copy(i) for i in self.indices]
        return Tensor(newindices, self.name, self.sym)


def permute(t, p):
    name = str(t.name)
    indices = [t.indices[i] for i in p]
    newt = Tensor(indices, name, sym=t.sym)
    return newt


class Sigma(object):
    """
    Class representing a sum over an index.

    Attributes:
        idx (Idx): Summed index
    """
    def __init__(self, idx):
        self.idx = idx

    def __eq__(self, other):
        if isinstance(other, Sigma):
            return self.idx == other.idx
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.idx < other.idx

    def __gt__(self, other):
        return self.idx > other.idx

    def __le__(self, other):
        return self.idx <= other.idx

    def __ge__(self, other):
        return self.idx >= other.idx

    def __hash__(self):
        return hash(str(self.idx))

    def __repr__(self):
        return "\\sum_{" + str(self.idx.index) + "}"

    def _inc(self, i):
        return Sigma(Idx(self.idx.index + i, self.idx.space))

    def _print_str(self, imap):
        return "\\sum_{" + imap[self.idx] + "}"

    def copy(self):
        return Sigma(idx_copy(self.idx))


class Delta(object):
    """
    Class reprenting a delta function.

    Attributes:
        i1 (Idx): First index
        i2 (Idx): Second index
    """
    def __init__(self, i1, i2):
        assert i1.space == i2.space
        self.i1 = i1
        self.i2 = i2

    def __eq__(self, other):
        if isinstance(other, Delta):
            return (self.i1 == other.i1 and self.i2 == other.i2) or (
                self.i1 == other.i2 and self.i2 == other.i1)
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(''.join(sorted(str(self.i1.index) + str(self.i2.index))))

    def __repr__(self):
        istr = str(self.i1.index) + "," + str(self.i2.index)
        return "\\delta_{" + istr + "}"

    def _inc(self, i):
        return Delta(
            Idx(self.i1.index + i, self.i1.space),
            Idx(self.i2.index + i, self.i2.space))

    def _print_str(self, imap):
        return "\\delta_{" + imap[self.i1] + imap[self.i2] + "}"

    def copy(self):
        i1 = idx_copy(self.i1)
        i2 = idx_copy(self.i2)
        return Delta(i1, i2)


def tensor_from_delta(d):
    sym = TensorSym([(0, 1), (1, 0)], [1, 1])
    t = Tensor([d.i1, d.i2], "delta", sym=sym)
    return t


def is_normal_ordered(operators, occ):
    fa = None
    for i, op in enumerate(operators):
        if fa is None and (not op.qp_creation(occ)):
            fa = i
        if fa is not None and op.qp_creation(occ):
            return False
    return True


def normal_ordered(operators, occ=None, sign=1):
    if is_normal_ordered(operators, occ):
        return (operators, sign)
    fa = None
    swap = None
    for i, op in enumerate(operators):
        assert isinstance(op, FOperator)
        if fa is None and (not op.qp_creation(occ)):
            fa = i
        if fa is not None and op.qp_creation(occ):
            swap = i
            break
    assert swap is not None
    newops = operators[:fa] + [operators[swap]]\
        + operators[fa:swap] + operators[swap + 1:]
    newsign = 1 if len(operators[fa:swap]) % 2 == 0 else -1
    sign = sign*newsign
    return normal_ordered(newops, sign=sign)
