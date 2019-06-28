class Idx(object):
    def __init__(self, index, space):
        self.index = index
        self.space = space

    def __repr__(self):
        return str(self.index) + "(" + self.space + ")"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.index == other.index\
                and self.space == other.space

    def __ne__(self, other):
        return not self.__eq__(other)

