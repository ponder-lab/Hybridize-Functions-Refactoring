def g(a_list):
    a_list[0] = 1


def f():
    my_list = [10]
    g(my_list)


f()
