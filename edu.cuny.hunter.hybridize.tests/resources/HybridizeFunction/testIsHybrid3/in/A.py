def function(func):
    """
    A custom decorator that does nothing.
    """
    return func


@function  # This is not the "function" from TensorFlow.
def func1():
    pass


if __name__ == "__main__":
    func1()
