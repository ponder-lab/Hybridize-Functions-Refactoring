def dummy_func(): # Not hybrid because there is no decorator.
    pass

@debug # Also not hybrid. This time, there is a decorator, it's not tf.function.
def dummy_func2():
    pass
