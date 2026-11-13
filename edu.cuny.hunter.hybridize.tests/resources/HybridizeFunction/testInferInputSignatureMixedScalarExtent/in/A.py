import tensorflow as tf


def sized(t):
    return tf.reduce_sum(t)


def counted(t):
    return tf.reduce_sum(t)


def fixed(t):
    return tf.reduce_sum(t)


xs = [1.0, 2.0, 3.0]

for empty in (True, False):
    if empty:
        n = 0
    else:
        n = len(xs)

    print(sized(tf.zeros([n])))

i = 0

while i < 3:
    print(counted(tf.zeros([i])))
    i += 1

print(fixed(tf.zeros([2])))
