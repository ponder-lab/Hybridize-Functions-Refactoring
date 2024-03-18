# Test https://github.com/wala/ML/issues/65.

from tensorflow import ones
from tensorflow import Tensor


def f(a):
    assert isinstance(a, Tensor)


f(ones([1, 2]))
