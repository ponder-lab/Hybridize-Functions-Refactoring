# Test https://github.com/wala/ML/issues/65.

from tensorflow import *


def f(a):
    assert isinstance(a, Tensor)


def g(a):
    assert not isinstance(a, Tensor)


f(ones([1, 2]))
g(5)
