# Test https://github.com/wala/ML/issues/65.

from tensorflow import *


def g(a):
    assert isinstance(a, Tensor)


def f(a):
    assert not isinstance(a, Tensor)
    g(ones([1, 2]))


f(5)
