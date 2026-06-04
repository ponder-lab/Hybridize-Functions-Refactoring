# Shape B (#578): `from tensorflow import function` plus a later `import tensorflow as tf`. The qualified `tf.` import makes
# the whole signature reachable, so the scan must not stop at the bare `from`-import and skip emission — it emits a
# `tf.`-qualified signature.
from tensorflow import function
import tensorflow as tf


def f(x):
    return x + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
