# Copyright (c) 2020 Alec White
# Licensed under the MIT License (see LICENSE for details)
from copy import copy


class Idx(object):
    """
    Index class

    Attributes:
        index (int): Integer labelling an index as unique
        space (str): Name of the index space
        fermion (bool): Index of fermiond space?
    """
    def __init__(self, index, space, fermion=True):
        self.index = index
        self.space = space
        self.fermion = fermion

    def __repr__(self):
        return str(self.index) + "(" + self.space + ")"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.index == other.index\
                and self.space == other.space

    def __ne__(self, other):
        return not self.__eq__(other)


def idx_copy(idx):
    """
    Copy an index by making a deep copy of the index (int) and
    fermion (bool) variables and copying the reference to the space.
    """
    return Idx(copy(idx.index), idx.space, bool(idx.fermion))

def is_occupied(idx, occ=None):
    if occ is None:
        return 'o' in idx.space
    else:
        return idx.space in occ
