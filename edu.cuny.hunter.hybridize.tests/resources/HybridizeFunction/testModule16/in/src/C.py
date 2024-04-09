# Test https://github.com/wala/ML/issues/163.

from tensorflow import Tensor
from src.B import C


class D(C):

    def g(self, a):
        assert isinstance(a, Tensor)
