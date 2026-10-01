# Direct-import control for `testReconfigureStarImport` (issue 1018): the same method, with `tf` imported by the file itself, is
# reconfigured with the signature qualified under `tf`, and `experimental_relax_shapes=True` is kept (#982).
import tensorflow as tf


class M:

    @tf.function(experimental_relax_shapes=True)
    def step(self, inputs, targets):
        return inputs + targets


if __name__ == "__main__":
    m = M()
    m.step(tf.constant([[1, 2]]), tf.constant([[3, 4]]))
