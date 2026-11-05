import tensorflow as tf


class Box:
    def ident(self, v):
        return v


def counted(x, flag, box):
    f = len if flag else box.ident
    return tf.range(f(x))


box = Box()
print(counted(tf.constant(3), False, box))
print(counted(tf.ones([4]), True, box))
