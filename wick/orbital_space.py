class OrbitalSpace(object):
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)

#def concatenate(space1, space2, name):
#    return orbital_space(name, space1.size() + space2.size())

#def compare(space1, space2):
#    if space1.name == space2.name:
#        return True
#    else:
#        return False
