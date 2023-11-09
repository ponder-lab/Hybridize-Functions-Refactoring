# From https://www.tensorflow.org/guide/function#executing_python_side_effects.


def f(x):
    x = 5  # no side-effect as `x` is a local variable..


f(1)
f(1)
f(2)
