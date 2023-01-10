# WICK
Symbolic manipulation of operator strings for quantum chemistry appliciations

[![Tests](https://github.com/awhite862/wick/workflows/Tests/badge.svg)](https://github.com/awhite862/wick/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/awhite862/wick/branch/master/graph/badge.svg)](https://codecov.io/gh/awhite862/wick)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/awhite862/wick/master/LICENSE)

## Features
WICK is a pure python library for applying Wick's theorem, manipulating, and simplifying arbitrary strings of second-quantized operators.
Some features include:
  - Fermion and Boson operators
  - LaTeX output
  - Numpy einsum output

## Examples
see the [examples](../master/examples)

## Tests
The provided tests should guarantee that all covered code is internally consistent.
The tests can be run as follows:
  - Individually from the `wick/tests` subdirectory
  - All at once by running `python test_suites.py` from `wick/tests`
  - All at once by running `python test.py`

The shell scripts in the [examples](../master/examples)
directory can be run to compare output text to the expected results in the
`examples/*.out` files. These results have been checked by hand against the known
equations.
