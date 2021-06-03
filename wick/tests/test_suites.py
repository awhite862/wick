import unittest
import test_aterm
import test_convenience
import test_expression
import test_full
import test_idx
import test_operators
import test_sc_rules
import test_term
import test_term_map
import test_test
import test_wick


def run_suite():
    suite = unittest.TestSuite()

    suite.addTest(test_test.TestTest("test_framework"))

    suite.addTest(test_aterm.ATermTest("test_mul"))
    suite.addTest(test_aterm.ATermTest("test_merge_external"))
    suite.addTest(test_aterm.ATermTest("test_term_map"))
    suite.addTest(test_aterm.ATermTest("test_tensor_sort"))
    suite.addTest(test_aterm.ATermTest("test_connected"))
    suite.addTest(test_aterm.ATermTest("test_reducible"))
    suite.addTest(test_aterm.ATermTest("test_eq"))
    suite.addTest(test_aterm.ATermTest("test_string"))

    suite.addTest(test_convenience.ConvenienceTest("testE1"))
    suite.addTest(test_convenience.ConvenienceTest("testE2"))
    suite.addTest(test_convenience.ConvenienceTest("testEip1"))
    suite.addTest(test_convenience.ConvenienceTest("testEip2"))
    suite.addTest(test_convenience.ConvenienceTest("testEdip1"))
    suite.addTest(test_convenience.ConvenienceTest("testEea1"))
    suite.addTest(test_convenience.ConvenienceTest("testEea2"))
    suite.addTest(test_convenience.ConvenienceTest("testEdea1"))
    suite.addTest(test_convenience.ConvenienceTest("testP1"))
    suite.addTest(test_convenience.ConvenienceTest("testP2"))
    suite.addTest(test_convenience.ConvenienceTest("testP1E1"))
    suite.addTest(test_convenience.ConvenienceTest("testP1Eip1"))
    suite.addTest(test_convenience.ConvenienceTest("testP1Eea1"))
    suite.addTest(test_convenience.ConvenienceTest("testP2E1"))
    suite.addTest(test_convenience.ConvenienceTest("testP1op"))
    suite.addTest(test_convenience.ConvenienceTest("testP2op"))
    suite.addTest(test_convenience.ConvenienceTest("testEP11op"))

    suite.addTest(test_expression.ExpressionTest("test_resolve0"))
    suite.addTest(test_expression.ExpressionTest("test_resolve1"))
    suite.addTest(test_expression.ExpressionTest("test_resolve2"))
    suite.addTest(test_expression.ExpressionTest("test_resolve3"))
    suite.addTest(test_expression.ExpressionTest("test_resolve_chain"))
    suite.addTest(test_expression.ExpressionTest("test_str"))

    suite.addTest(test_full.FullTest("test_ccsd_T1"))
    suite.addTest(test_full.FullTest("test_p2"))

    suite.addTest(test_idx.IdxTest("test_idx"))
    suite.addTest(test_idx.IdxTest("test_idx_str"))
    suite.addTest(test_idx.IdxTest("test_idx_occ"))
    suite.addTest(test_idx.IdxTest("test_idx_eq"))

    suite.addTest(test_operators.OperatorTest("test_foperator"))
    suite.addTest(test_operators.OperatorTest("test_boperator"))
    suite.addTest(test_operators.OperatorTest("test_projector"))
    suite.addTest(test_operators.OperatorTest("test_tensor"))
    suite.addTest(test_operators.OperatorTest("test_sigma"))
    suite.addTest(test_operators.OperatorTest("test_delta"))
    suite.addTest(test_operators.OperatorTest("test_dagger"))
    suite.addTest(test_operators.OperatorTest("test_string"))
    suite.addTest(test_operators.OperatorTest("test_inc"))

    suite.addTest(test_sc_rules.SCRulesTest("test_0d0"))
    suite.addTest(test_sc_rules.SCRulesTest("test_0d1"))
    suite.addTest(test_sc_rules.SCRulesTest("test_0d2"))
    suite.addTest(test_sc_rules.SCRulesTest("test_1d0a"))
    suite.addTest(test_sc_rules.SCRulesTest("test_1d1a"))
    suite.addTest(test_sc_rules.SCRulesTest("test_1d2a"))
    suite.addTest(test_sc_rules.SCRulesTest("test_1d0b"))
    suite.addTest(test_sc_rules.SCRulesTest("test_1d1b"))
    suite.addTest(test_sc_rules.SCRulesTest("test_1d2b"))
    suite.addTest(test_sc_rules.SCRulesTest("test_2d1a"))
    suite.addTest(test_sc_rules.SCRulesTest("test_2d2a"))
    suite.addTest(test_sc_rules.SCRulesTest("test_2d1b"))
    suite.addTest(test_sc_rules.SCRulesTest("test_2d2b"))
    suite.addTest(test_sc_rules.SCRulesTest("test_2d1c"))
    suite.addTest(test_sc_rules.SCRulesTest("test_2d2c"))

    suite.addTest(test_term_map.TermMapTest("test_null"))
    suite.addTest(test_term_map.TermMapTest("test_label"))

    suite.addTest(test_term.TermTest("test_scalar_mul"))
    suite.addTest(test_term.TermTest("test_mul"))
    suite.addTest(test_term.TermTest("test_mul2"))
    suite.addTest(test_term.TermTest("test_ilist"))

    suite.addTest(test_wick.WickTest("test_valid_contraction"))
    suite.addTest(test_wick.WickTest("test_pair_list"))
    suite.addTest(test_wick.WickTest("test_get_sign"))
    suite.addTest(test_wick.WickTest("test_split_operators"))
    suite.addTest(test_wick.WickTest("test_projector"))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(run_suite())
