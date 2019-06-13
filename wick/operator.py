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
    def __init__(self, indices, spaces, bk, name):
        self.indices = indices
        self.spaces = spaces
        self.bk = bk
        assert(len(bk) == len(spaces))
        assert(len(spaces) == len(indices))
        self.name = name

    def print_str(self):
        temp = self.name
        r = str()
        l = str()
        for i in range(len(self.bk)):
            if self.bk[i]:
                r = r + str(self.indices[i])
            else:
                l = l + str(self.indices[i])

        return self.name + "_{" + l + r + "}"


class Sigma(object):
    def __init__(self, index, space):
        self.index = index
        self.space = space

    def __eq__(self, other):
        return self.index == other.index and self.space == other.space

    def __ne__(self, other):
        return not self.__eq__(other)

    def print_str(self):
        return "\sum_{" + str(self.index) + "}"

class Delta(object):
    def __init__(self, i1, i2, s1, s2):
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
