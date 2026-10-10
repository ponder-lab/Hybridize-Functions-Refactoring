class Box:
    pass


def make():
    return Box()


def hub(n):
    if n > 0:
        left(n - 1)
        right(n - 1)
    return make()


def left(n):
    box = hub(n)
    box.value = 1
    return box


def right(n):
    box = hub(n)
    box.value = 2
    return box


left(2)
right(2)
