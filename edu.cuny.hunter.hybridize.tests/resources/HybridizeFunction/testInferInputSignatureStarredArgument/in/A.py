import tensorflow as tf


def t3(x, y):
    return x - y


def t4(x, y):
    return x - y


def t5(x, y):
    return x - y


t3(tf.ones([4]), *[tf.ones([4])])

t4(tf.ones([4]), tf.ones([4]))

t5(*[tf.ones([4])], y=tf.ones([4]))


class M:
    def m(self, x, y):
        return x + y

    def plain(self, x, y):
        return x + y

    def before_star(self, pair, y):
        a, b = pair
        return a + b + y


M().m(tf.ones([4]), *[tf.ones([4])])
M().plain(tf.ones([4]), tf.ones([4]))
M().before_star((tf.ones([4]), tf.ones([4])), *[tf.ones([4])])


class K:
    @staticmethod
    def sm_inst(x, y):
        return x + y

    @staticmethod
    def sm(x, y):
        return x + y


K().sm_inst(tf.ones([4]), *[tf.ones([4])])
K.sm(tf.ones([4]), tf.ones([4]))
