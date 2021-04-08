import unittest
import test_aterm
import test_idx
import test_operators
import test_term
import test_term_map
import test_test
import test_wick

def run_suite():
    suite = unittest.TestSuite()

    suite.addTest(test_test.TestTest("test_framework"))

    suite.addTest(test_aterm.ATermTest("test_connected"))
    suite.addTest(test_aterm.ATermTest("test_reducible"))

    suite.addTest(test_idx.IdxTest("test_idx"))
    suite.addTest(test_idx.IdxTest("test_idx_str"))
    suite.addTest(test_idx.IdxTest("test_idx_occ"))

    suite.addTest(test_operators.OperatorTest("test_operator"))
    suite.addTest(test_operators.OperatorTest("test_tensor"))
    suite.addTest(test_operators.OperatorTest("test_sigma"))
    suite.addTest(test_operators.OperatorTest("test_delta"))
    suite.addTest(test_operators.OperatorTest("test_dagger"))

    suite.addTest(test_term_map.TermMapTest("test_null"))
    suite.addTest(test_term_map.TermMapTest("test_label"))

    suite.addTest(test_term.TermTest("test_scalar_mul"))
    suite.addTest(test_term.TermTest("test_mul"))
    suite.addTest(test_term.TermTest("test_mul2"))
    suite.addTest(test_term.TermTest("test_ilist"))
    suite.addTest(test_term.TermTest("test_term_map"))

    suite.addTest(test_wick.WickTest("test_valid_contraction"))
    suite.addTest(test_wick.WickTest("test_pair_list"))
    suite.addTest(test_wick.WickTest("test_get_sign"))
    suite.addTest(test_wick.WickTest("test_split_operators"))
    suite.addTest(test_wick.WickTest("test_projector"))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(run_suite())
