class Operator(object):
    def __init__(self, index, space, ca):
        self.index = index
        self.space = space
        self.ca = ca

    def __eq__(self, other):
        return self.index == other.index and \
                self.space == other.space and \
                self.ca == other.ca

    def __ne__(self, other):
        return not self.__eq__(other)

    def print_str(self):
        if self.ca:
            return "a^{\dagger}_" + str(self.index)
        else:
            return "a_" + str(self.index)

class Tensor(object):
    def __init__(self, indices, spaces, name):
        self.indices = indices
        self.spaces = spaces
        assert(len(spaces) == len(indices))
        self.name = name

    def __eq__(self, other):
        return self.indices == other.indices \
                and self.spaces == other.spaces \
                and self.name == other.name

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        ss = str(self.name)
        for i in self.indices:
            ss += i
        for sp in self.spaces:
            ss += str(sp)
        return hash(ss)

    def print_str(self):
        temp = self.name
        s = str()
        for idx in self.indices:
            s += str(idx)

        return self.name + "_{" + s + "}"


class Sigma(object):
    def __init__(self, index, space):
        self.index = index
        self.space = space

    def __eq__(self, other):
        return self.index == other.index and self.space == other.space

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(''.join(self.index + str(self.space)))

    def __repr__(self):
        return "\sum_{" + str(self.index) + "}"

class Delta(object):
    def __init__(self, i1, i2, s1, s2):
        assert(s1 == s2)
        self.i1 = i1
        self.i2 = i2
        self.s1 = s1
        self.s2 = s2

    def __eq__(self, other):
        return (self.i1 == other.i1 and
                self.i2 == other.i2 and
                self.s1 == other.s1 and
                self.s2 == other.s2) or (
                self.i1 == other.i2 and
                self.i2 == other.i1 and 
                self.s1 == other.s2 and
                self.s2 == other.s1)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(''.join(sorted(self.i1 + self.i2)))

    def print_str(self):
        return "\delta_{" + self.i1 + "," + self.i2 + "}"
