import tensorflow as tf


# Positional `func=None` plus keyword `autograph=False`.
# Tests #108: a single decorator that mixes positional and keyword args should
# have BOTH parameter-existence flags set (`func` from position 0,
# `autograph` from the keyword).
@tf.function(None, autograph=False)
def func(x):
    return x


if __name__ == "__main__":
    number = tf.constant([1.0, 1.0])
    func(number)
