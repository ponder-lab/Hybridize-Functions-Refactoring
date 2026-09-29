# A project whose source folder, in/, leaves a script with an import, extra/B.py, uncovered (#990).
import tensorflow as tf


def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(1.0))
