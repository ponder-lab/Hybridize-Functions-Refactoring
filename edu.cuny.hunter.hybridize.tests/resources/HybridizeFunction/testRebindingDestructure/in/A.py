import tensorflow as tf


def consume_plain_second(t):
    pass


def consume_rebound_second(t):
    pass


# Control: the left-hand side does not mention the name being destructured.
def plain(pair):
    first, second = pair
    consume_plain_second(second)


# Field 0's target rebinds the very name on the right, which is
# `gpt-2-tensorflow2.0`'s `train_dataset, test_dataset = train_dataset`.
def rebinding(pair):
    pair, second = pair
    consume_rebound_second(second)


a = tf.ones((2, 3))
b = tf.ones((4, 5))

plain((a, b))
rebinding((a, b))
