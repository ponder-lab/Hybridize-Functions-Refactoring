# A broader supplied signature on a method of a user class with its own save method (#808). A save call on an object other than the
# SavedModel module may be a Keras model saving itself, so the method is possibly exported, and the narrowing is declined as undetermined.
import tensorflow as tf


class Trainer:
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def step(self, t):
        return t + 1

    def save(self, path):
        pass


if __name__ == "__main__":
    trainer = Trainer()
    trainer.step(tf.constant(2.0))
    trainer.save("checkpoint")
