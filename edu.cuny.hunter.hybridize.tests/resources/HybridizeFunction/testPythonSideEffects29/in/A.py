a = 10


def g():
    global a
    a = 1


def f():
    g()


assert a == 10
f()
assert a == 1, "Function f() calls g(), function g() modifies a global variable."
