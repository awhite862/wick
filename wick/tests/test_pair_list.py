import unittest

from wick.wick import pair_list

class PairListTest(unittest.TestCase):
    def test_p2(self):
        lst = [0,1]
        plist = pair_list(lst)
        ref = [[(0,1),],]
        self.assertTrue(ref == plist)

    def test_p4(self):
        lst = [0,1,2,3]
        plist = pair_list(lst)
        ref = [[(0,1),(2,3)],
               [(0,2),(1,3)],
               [(0,3),(1,2)]]
        self.assertTrue(ref == plist)

    def test_p6(self):
        lst = [0,1,2,3,4,5]
        plist = pair_list(lst)
        ref = [[(0,1),(2,3),(4,5)],
               [(0,1),(2,4),(3,5)],
               [(0,1),(2,5),(3,4)],
               [(0,2),(1,3),(4,5)],
               [(0,2),(1,4),(3,5)],
               [(0,2),(1,5),(3,4)],
               [(0,3),(1,2),(4,5)],
               [(0,3),(1,4),(2,5)],
               [(0,3),(1,5),(2,4)],
               [(0,4),(1,2),(3,5)],
               [(0,4),(1,3),(2,5)],
               [(0,4),(1,5),(2,3)],
               [(0,5),(1,2),(3,4)],
               [(0,5),(1,3),(2,4)],
               [(0,5),(1,4),(2,3)],]
        self.assertTrue(ref == plist)

if __name__ == '__main__':
    unittest.main()
