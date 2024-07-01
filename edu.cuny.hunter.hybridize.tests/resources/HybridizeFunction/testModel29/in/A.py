# Test https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/229.

import tensorflow as tf


class EmptyClass:
    pass


class EmptyClass2(tf.keras.Model):
    pass


class SequentialModel(EmptyClass, EmptyClass2):

    def __init__(self, **kwargs):
        super(SequentialModel, self).__init__(**kwargs)

        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))

        # Add a lot of small layers
        num_layers = 100
        self.my_layers = [
            tf.keras.layers.Dense(64, activation="relu") for n in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense_2 = tf.keras.layers.Dense(10)

    def __call__(self, x):
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    input_data = tf.random.uniform([20, 28, 28])
    model = SequentialModel()
    result = model(5)
