# Test https://github.com/wala/ML/issues/65.

from tensorflow import ones, Tensor, function


@function
def g(a):
    assert isinstance(a, Tensor)


@function
def f(a):
    assert isinstance(a, Tensor)
    g(a)


f(ones([1, 2]))
