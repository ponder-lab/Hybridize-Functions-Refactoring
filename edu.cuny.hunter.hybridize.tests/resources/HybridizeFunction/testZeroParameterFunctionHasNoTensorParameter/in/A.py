# Regression fixture for #498. A function with no parameters trivially has no tensor
# parameter. `Function.getHasTensorParameter() == FALSE` (the classification loop has
# nothing to iterate over).


def f():
    pass


f()
