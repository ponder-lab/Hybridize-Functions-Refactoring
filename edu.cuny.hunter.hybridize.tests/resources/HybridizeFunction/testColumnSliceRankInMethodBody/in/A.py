import tensorflow as tf


class Feeder:
    def run(self, table):
        return step(table[:, 0])


def step(x):
    return x


def step_control(x):
    return x


table = tf.ones((30, 2), dtype=tf.int32)

Feeder().run(table)
step_control(table[:, 0])
