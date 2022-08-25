def dummy_decorator(func):
    """
    A custom decorator that does nothing.
    """
    return func

def dummy_func(): # Not hybrid because there is no decorator.
    pass

@dummy_decorator # Also not hybrid. This time, there is a decorator, it's not tf.function.
def dummy_func2():
    pass

if __name__ == '__main__':
    dummy_func()
    dummy_func2()
