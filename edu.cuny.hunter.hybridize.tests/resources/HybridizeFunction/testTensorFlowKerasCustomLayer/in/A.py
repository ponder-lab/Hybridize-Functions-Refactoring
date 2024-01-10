# From https://github.com/ponder-lab/samples/blob/bc64c0421f9a8a040d20f2c7a72533b58b938b5e/tensorflow_keras_custom_layer/tensorflow_keras_custom_layer.py.
#!/usr/bin/env python

import sys
import time
import random

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class MyConvolution2D(keras.layers.Layer):

    def __init__(self, num_och, kernel_size, padding, **kwargs):
        super(MyConvolution2D, self).__init__(**kwargs)
        self.num_och = num_och
        self.kernel_size = kernel_size
        self.padding = padding

    # !!!! Need get_config if saving to file !!!!
    def get_config(self):
        config = super(MyConvolution2D, self).get_config()
        print(config)
        config.update({
            "num_och": self.num_och,
            "kernel_size": self.kernel_size,
            "padding": self.padding
            })
        return config

    def build(self, input_shape):
        # !!!! Need to set names if saving to file !!!!
        self.w = self.add_weight(shape=(self.num_och,
                                        self.kernel_size[0] * self.kernel_size[1] * input_shape[-1]),
                                 initializer="random_normal", name="my_weight",
                                 trainable=True)
        self.b = self.add_weight(shape=(self.num_och,),
                                 initializer="random_normal", name="my_bias",
                                 trainable=True)

    def call(self, inputs):
        # Get the 3x3 convolution patches
        data = tf.image.extract_patches(inputs, sizes=(1, 3, 3, 1), strides=(1, 1, 1, 1),
                                        rates=(1, 1, 1, 1), padding=self.padding)

        # Copy for each output channel
        data = tf.tile(data, (1, 1, 1, self.num_och))
        # !!!! has to pass tf.shape(data)[0] instead of data.shape[0] since called w/ None !!!!
        data = tf.reshape(data, (tf.shape(data)[0], data.shape[1], data.shape[2], self.num_och,
                                 self.kernel_size[0] * self.kernel_size[1] * inputs.shape[3]))

        # Multiply by weight and sum
        data *= self.w
        data = tf.reduce_sum(data, axis=-1)

        # Add bias
        data += self.b

        return data


def get_model(input_shape):
    input_node = keras.Input(input_shape)
    x = MyConvolution2D(64, (3, 3), padding="SAME")(input_node)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Flatten()(x)
    output_node = keras.layers.Dense(10, activation="softmax")(x)
    return keras.models.Model(input_node, output_node)


start = time.time()

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = (x_train.astype(np.float32) / 255.0)[..., tf.newaxis]
x_test = (x_test.astype(np.float32) / 255.0)[..., tf.newaxis]

model = get_model(x_train[0].shape)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

num_epochs = 1
batch_size = 100
validation_split = 0.2
model_file = "tmp.h5"

callbacks = [keras.callbacks.ModelCheckpoint(filepath=model_file, \
                                             save_best_only=True)]

model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          callbacks=callbacks, verbose=1, validation_split=validation_split)

# Need to pass custom layer to load_model
model = keras.models.load_model(model_file, custom_objects={"MyConvolution2D": MyConvolution2D})

score = model.evaluate(x_test, y_test, verbose=0)
print("Test score:", score[0])
print("Test accuracy:", score[1])

print("Elapsed time:", time.time() - start)
