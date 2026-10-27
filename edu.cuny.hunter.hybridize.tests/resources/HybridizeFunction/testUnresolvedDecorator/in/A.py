# Bound at run time only, so the decorator has no static definition.
globals()["unregistered_decorator"] = lambda func: func


@unregistered_decorator
def f(x):
    return x


print(f(1))
