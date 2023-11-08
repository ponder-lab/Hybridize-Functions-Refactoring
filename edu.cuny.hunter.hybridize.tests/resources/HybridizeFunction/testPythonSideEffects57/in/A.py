def f():

    def g():
        return 5

    a = g()
    assert a == 5


if __name__ == "__main__":
    f()
