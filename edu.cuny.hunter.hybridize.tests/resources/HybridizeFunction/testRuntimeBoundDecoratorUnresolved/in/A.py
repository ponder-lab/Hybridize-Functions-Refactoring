import importlib

# Bind `runtime_bound_decorator` at run time, so this module has no definition or import of it.
globals()["runtime_bound_decorator"] = importlib.import_module(
    "B"
).runtime_bound_decorator


@runtime_bound_decorator
def f(x):
    return x


print(f(1))
