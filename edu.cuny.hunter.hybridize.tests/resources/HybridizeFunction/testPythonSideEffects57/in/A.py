def f():

    def g():
        return 5

    a = g()
    assert a == 5


f()
