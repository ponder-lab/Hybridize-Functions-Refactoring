# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1061

import numpy as np
import tensorflow as tf


def helper0(f, v):
    def inner():
        return v * 1.0

    return f() + inner()


def helper1(f, v):
    def inner():
        return v * 2.0

    return f() + inner()


def helper2(f, v):
    def inner():
        return v * 3.0

    return f() + inner()


def helper3(f, v):
    def inner():
        return v * 4.0

    return f() + inner()


def helper4(f, v):
    def inner():
        return v * 5.0

    return f() + inner()


def helper5(f, v):
    def inner():
        return v * 6.0

    return f() + inner()


def helper6(f, v):
    def inner():
        return v * 7.0

    return f() + inner()


def helper7(f, v):
    def inner():
        return v * 8.0

    return f() + inner()


def helper8(f, v):
    def inner():
        return v * 9.0

    return f() + inner()


def helper9(f, v):
    def inner():
        return v * 10.0

    return f() + inner()


def helper10(f, v):
    def inner():
        return v * 11.0

    return f() + inner()


def helper11(f, v):
    def inner():
        return v * 12.0

    return f() + inner()


def helper12(f, v):
    def inner():
        return v * 13.0

    return f() + inner()


def helper13(f, v):
    def inner():
        return v * 14.0

    return f() + inner()


def helper14(f, v):
    def inner():
        return v * 15.0

    return f() + inner()


def helper15(f, v):
    def inner():
        return v * 16.0

    return f() + inner()


def outer(x):
    a0 = lambda: helper0(lambda: x + 0.0, x)
    a1 = lambda: helper1(lambda: x + 1.0, x)
    a2 = lambda: helper2(lambda: x + 2.0, x)
    a3 = lambda: helper3(lambda: x + 3.0, x)
    a4 = lambda: helper4(lambda: x + 4.0, x)
    a5 = lambda: helper5(lambda: x + 5.0, x)
    a6 = lambda: helper6(lambda: x + 6.0, x)
    a7 = lambda: helper7(lambda: x + 7.0, x)
    a8 = lambda: helper8(lambda: x + 8.0, x)
    a9 = lambda: helper9(lambda: x + 9.0, x)
    a10 = lambda: helper10(lambda: x + 10.0, x)
    a11 = lambda: helper11(lambda: x + 11.0, x)
    a12 = lambda: helper12(lambda: x + 12.0, x)
    a13 = lambda: helper13(lambda: x + 13.0, x)
    a14 = lambda: helper14(lambda: x + 14.0, x)
    a15 = lambda: helper15(lambda: x + 15.0, x)
    return (
        a0()
        + a1()
        + a2()
        + a3()
        + a4()
        + a5()
        + a6()
        + a7()
        + a8()
        + a9()
        + a10()
        + a11()
        + a12()
        + a13()
        + a14()
        + a15()
    )


def outer_numpy(x):
    b = lambda: helper0(lambda: np.sum(x), x)
    return b() + outer(x)


t = tf.constant([1.0, 2.0])
outer(t)
outer_numpy(t)
