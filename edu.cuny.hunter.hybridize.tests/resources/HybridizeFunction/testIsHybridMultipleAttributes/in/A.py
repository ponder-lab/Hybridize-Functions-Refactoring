#!/usr/bin/python3

import tensorflow
import pytest
import sys

@tensorflow.autograph.experimental.do_not_convert
def dummy_fun():
    pass

@pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
def dummy_test(test_input, expected):
    pass

@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_function():
    pass

if __name__ == '__main__':
    dummy_fun()
    dummy_test("1", "1")
    test_function()
