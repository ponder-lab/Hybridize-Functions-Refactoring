# Test https://github.com/wala/ML/issues/163.

from tensorflow import Tensor
from B import C


class D(C):

    def f(self, a):
        assert isinstance(a, Tensor)
