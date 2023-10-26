# From https://stackoverflow.com/questions/5753597/is-it-pythonic-to-use-list-comprehensions-for-just-side-effects

my_list = [10]


def h():
    pass


def fun_with_side_effects(y):
    my_list[0] = 1
    return y ** 2


def g():
    squares = [fun_with_side_effects(x) for x in range(10)]


def f():
    g()


f()
h()
