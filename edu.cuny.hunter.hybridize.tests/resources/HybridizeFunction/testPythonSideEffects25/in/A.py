# From https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions


def g():
    squares = []
    for x in range(10):
        squares.append(x ** 2)


def f():
    g()


f()