def f(x):
    def g(y, z=[]):  # The default list is created in `f` and bound with `g`.
        return y

    return g(x)


f(1)
