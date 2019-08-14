from .index import Idx

class FOperator(object):
    """
    Fermion creation/annihilation operators
    
    idx (Idx): Index of operator
    ca (Bool): Creation operator?
    """
    def __init__(self, idx, ca):
        self.idx = idx
        self.ca = ca

    def __eq__(self, other):
        return self.idx == other.idx and \
                self.ca == other.ca

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        if self.ca:
            return "a^{\dagger}_" + str(self.idx)
        else:
            return "a_" + str(self.idx)

    def _inc(self, i):
        """Increment indices"""
        return FOperator(Idx(self.idx.index + i, self.idx.space), self.ca)

    def _print_str(self, imap):
        if self.ca:
            return "a^{\dagger}_" + imap[self.idx]
        else:
            return "a_" + imap[self.idx]

    def qp_creation(self, occ=None):
        if (not self.idx.is_occupied(occ=occ)) and self.ca:
            return True
        elif (self.idx.is_occupied(occ=occ)) and not self.ca:
            return True
        else:
            return False

    def qp_anihilation(self, occ=None):
        return not self.qp_creation(occ=occ)

class TensorSym(object):
    """
    Representation of tensor permutational symmetry

    plist (lsit): List of tuples representing permutations of indices
    signs (list): List of signs representing the signes of each permutation
    tlist (list): List of permutation, sign pairs
    """
    def __init__(self, plist, signs):
        self.plist = plist
        self.signs = signs
        self.tlist = [(p,s) for p,s in zip(plist,signs)]

class Tensor(object):
    """
    Tensor

    indices (list): List of indices
    name (str): Name of the tensor
    sym (TensorSym): Permutational symmetry of tensor
    """
    def __init__(self, indices, name, sym=None):
        self.indices = indices
        self.name = name
        if sym is None:
            self.sym = TensorSym([tuple([i for i in range(len(indices))])], [1.0])
        else:
            self.sym = sym

    def __eq__(self, other):
        return self.indices == other.indices \
                and self.name == other.name

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        ss = str(self.name)
        for i in self.indices:
            ss += str(i)
        return hash(ss)

    def __repr__(self):
        temp = self.name
        s = str()
        for idx in self.indices:
            s += str(idx.index)

        return self.name + "_{" + s + "}"

    def _inc(self, i):
        indices = [Idx(ii.index + i, ii.space) for ii in self.indices]
        return Tensor(indices, self.name, sym=self.sym)

    def ilist(self):
        ilist = []
        for idx in self.indices:
            if idx not in ilist: ilist.append(idx)
        return ilist

    def _print_str(self, imap):
        temp = self.name
        if len(temp) == 0: return str()
        s = str()
        for idx in self.indices:
            s += imap[idx]

        return self.name + "_{" + s + "}"

def permute(t, p):
    name = str(t.name)
    indices = [t.indices[i] for i in p]
    newt = Tensor(indices, name, sym=t.sym)
    return newt

class Sigma(object):
    def __init__(self, idx):
        self.idx = idx

    def __eq__(self, other):
        return self.idx == other.idx

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self.idx))

    def __repr__(self):
        return "\sum_{" + str(self.idx.index) + "}"

    def _inc(self, i):
        return Sigma(Idx(self.idx.index + i, self.idx.space))

    def _print_str(self, imap):
        return "\sum_{" + imap[self.idx] + "}"

class Delta(object):
    def __init__(self, i1, i2):
        assert(i1.space == i2.space)
        self.i1 = i1
        self.i2 = i2

    def __eq__(self, other):
        return (self.i1 == other.i1 and
                self.i2 == other.i2) or (
                self.i1 == other.i2 and
                self.i2 == other.i1)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(''.join(sorted(str(self.i1.index) + str(self.i2.index))))

    def __repr__(self):
        return "\delta_{" + str(self.i1.index) + "," + str(self.i2.index) + "}"

    def _inc(self, i):
        return Delta(Idx(self.i1.index + i, self.i1.space), Idx(self.i2.index + 1, self.i2.space))

    def _print_str(self, imap):
        return "\delta_{" + imap[self.i1] + imap[self.i2] + "}"
