# From https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions


def g():
    squares = list(map(lambda x: x ** 2, range(10)))


def f():
    g()


f()
