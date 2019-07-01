import unittest
import test_operators
import test_term
import test_test

def run_suite():
    suite = unittest.TestSuite()

    suite.addTest(test_test.TestTest("test_framework"))

    suite.addTest(test_operators.OperatorTest("test_operator"))
    suite.addTest(test_operators.OperatorTest("test_tensor"))
    suite.addTest(test_operators.OperatorTest("test_sigma"))
    suite.addTest(test_operators.OperatorTest("test_delta"))

    suite.addTest(test_term.TermTest("test_scalar_mul"))
    suite.addTest(test_term.TermTest("test_mul"))
    suite.addTest(test_term.TermTest("test_mul2"))
    suite.addTest(test_term.TermTest("test_term_map"))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(run_suite())
