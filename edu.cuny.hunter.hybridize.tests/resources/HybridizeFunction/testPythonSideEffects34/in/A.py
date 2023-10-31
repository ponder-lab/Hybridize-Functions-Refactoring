def g(a):
    a = 1


def f():
    x = 10
    assert x == 10
    g(x)
    assert x == 10


f()
