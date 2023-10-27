a = 10


def h():
    global a
    a = 1


def g():
    pass


def f():
    g()


assert a == 10
f()
assert a == 10
h()
assert a == 1
