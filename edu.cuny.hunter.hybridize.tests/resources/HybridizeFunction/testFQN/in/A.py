import tensorflow as tf

# func
def func():
    pass

# func1
def func1():
    # func1.func2
    def func2():
        pass
    pass

class Class1:
    # Class1.func_class1
    def func_class1():
        pass
    class Class2:
        # Class1.Class2.func_class2
        def func_class2():
            pass
