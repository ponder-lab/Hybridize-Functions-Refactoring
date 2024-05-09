from B import g
from B import C


class D:
    pass


def f():
    g()
    D()
    C()


f()
