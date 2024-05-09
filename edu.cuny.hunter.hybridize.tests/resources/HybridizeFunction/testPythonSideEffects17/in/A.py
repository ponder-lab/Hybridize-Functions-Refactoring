# From https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions

my_list = [10]


def fun_with_side_effects(y):
    my_list[0] = 1
    return y**2


def f():
    squares = list(map(lambda x: fun_with_side_effects(x), range(10)))


f()
