# WICK
Symbolic manipulation of operator strings for quantum chemistry appliciations

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
