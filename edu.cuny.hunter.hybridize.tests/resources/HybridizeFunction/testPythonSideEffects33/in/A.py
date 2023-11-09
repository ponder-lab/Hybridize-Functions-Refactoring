def g(a_list):
    return a_list[0]


def f():
    my_list = [10]
    assert my_list[0] == 10
    g(my_list)
    assert my_list[0] == 10


f()
