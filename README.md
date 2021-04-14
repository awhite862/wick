# WICK
Symbolic manipulation of operator strings for quantum chemistry appliciations

[![Build](https://github.com/awhite862/wick/workflows/Build/badge.svg)](https://github.com/awhite862/wick/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/awhite862/wick/branch/master/graph/badge.svg)](https://codecov.io/gh/awhite862/wick)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://mit-license.org)

## Features
WICK is a pure python library for applying Wick's theorem, manipulating, and simplifying arbitrary strings of second-quantized operators.
Some features include:
  - Fermion and Boson operators
  - LaTeX output
  - Numpy einsum output

## Examples
see the [examples](../master/examples)

## Tests
We don't currently have very good test coverage for this library.
In the meantime, the shell scripts in the [examples](../master/examples)
directory can be run to compare output text to the expected results in the
`examples/*.out` files.

The tests we do have can be run as follows:
  - Individually from the `wick/tests` subdirectory
  - All at once by running `python test_suites.py` from `wick/tests`
  - All at once by running `python -m unittest test.py`
