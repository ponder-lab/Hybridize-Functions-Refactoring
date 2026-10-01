# Issue 1018: two already-hybrid methods in a file that reaches TensorFlow only through a star import, with signatures needing
# different dtypes. A single injected import carries both dtypes, and neither method injects a second one.
from B import *


class M:

    @tf.function(experimental_relax_shapes=True)
    def step(self, inputs):
        return inputs + 1

    @tf.function
    def evaluate(self, inputs):
        return inputs * 2


if __name__ == "__main__":
    m = M()
    m.step(tf.constant([[1, 2]]))
    m.evaluate(tf.constant([1.0, 2.0, 3.0]))
