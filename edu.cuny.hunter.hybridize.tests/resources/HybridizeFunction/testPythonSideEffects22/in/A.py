# From https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions


def g():
    squares = [x ** 2 for x in range(10)]


def f():
    g()


f()
