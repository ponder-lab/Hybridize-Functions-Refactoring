from tensorflow import Tensor


class C:

    def f(a):
        assert isinstance(a, Tensor)
