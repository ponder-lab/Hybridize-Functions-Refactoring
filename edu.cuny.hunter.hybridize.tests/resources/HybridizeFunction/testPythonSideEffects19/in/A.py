def g():
    my_list = [10]
    my_list[0] = 1


def f():
    g()


f()
