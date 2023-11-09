a = [10]


def g():
    a[0] = 1


def f():
    g()


assert a == [10]
f()
assert a == [1]
