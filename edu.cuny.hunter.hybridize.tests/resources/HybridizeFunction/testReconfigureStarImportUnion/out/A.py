# Issue 1018: two already-hybrid methods in a file that reaches TensorFlow only through a star import, with signatures needing
# different dtypes. A single injected import carries both dtypes, and neither method injects a second one.
from B import *
from tensorflow import function, TensorSpec, float32, int32


class M:

    @tf.function(experimental_relax_shapes=True, input_signature=[TensorSpec(shape=(1, 2), dtype=int32)])
    def step(self, inputs):
        return inputs + 1

    @tf.function(input_signature=[TensorSpec(shape=(3,), dtype=float32)])
    def evaluate(self, inputs):
        return inputs * 2


if __name__ == "__main__":
    m = M()
    m.step(tf.constant([[1, 2]]))
    m.evaluate(tf.constant([1.0, 2.0, 3.0]))
