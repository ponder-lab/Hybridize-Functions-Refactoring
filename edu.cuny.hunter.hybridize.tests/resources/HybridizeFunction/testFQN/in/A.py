import tensorflow as tf

# func
def func():
    pass

# func1
def func1():
    # func1.func2
    def func2():
        pass
    func2()

class Class1:
    # Class1.func_class1
    def func_class1():
        pass
    # Class1.func_class3
    def func_class3(self):
        pass
    class Class2:
        # Class1.Class2.func_class2
        def func_class2():
            pass
        # Class1.Class2.func_class4
        def func_class4(self):
            pass

if __name__ == "__main__":
    func()
    func1()
    Class1.func_class1()
    c = Class1()
    c.func_class3()
    Class1.Class2.func_class2()
    c2 = Class1.Class2()
    c2.func_class4()
