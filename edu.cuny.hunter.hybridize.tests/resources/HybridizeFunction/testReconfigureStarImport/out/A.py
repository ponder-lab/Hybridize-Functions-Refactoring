# Issue 1018: an already-hybrid method whose file reaches TensorFlow only through a star import. The file has no TensorFlow import
# of its own, so one is injected, and the inferred signature is added beside `experimental_relax_shapes=True`, which is kept (#982).
from B import *
from tensorflow import function, TensorSpec, int32


class M:

    @tf.function(experimental_relax_shapes=True, input_signature=[TensorSpec(shape=(1, 2), dtype=int32), TensorSpec(shape=(1, 2), dtype=int32)])
    def step(self, inputs, targets):
        return inputs + targets


if __name__ == "__main__":
    m = M()
    m.step(tf.constant([[1, 2]]), tf.constant([[3, 4]]))
