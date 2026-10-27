# Issue 1018: an already-hybrid method whose file reaches TensorFlow only through a star import. The file has no TensorFlow import
# of its own, so one is injected, and the inferred signature is added beside `experimental_relax_shapes=True`, which is kept (#982).
from B import *


class M:

    @tf.function(experimental_relax_shapes=True)
    def step(self, inputs, targets):
        return inputs + targets


if __name__ == "__main__":
    m = M()
    m.step(tf.constant([[1, 2]]), tf.constant([[3, 4]]))
