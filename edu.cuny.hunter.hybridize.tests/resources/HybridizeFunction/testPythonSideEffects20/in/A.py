a = 10


def g():
    global a
    a = a + 1


def f():
    g()


f()
