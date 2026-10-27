import tensorflow as tf

# Bound at run time only, so the decorator has no static definition.
globals()["unregistered_decorator"] = lambda func: func


@tf.function
@unregistered_decorator
def f(x):
    return x


class C:
    @unregistered_decorator
    def m(self, x):
        return x


print(f(tf.constant(1)))
print(C().m(tf.constant(1)))
