# From https://github.com/ponder-lab/samples/blob/39f7644391e664244b45c90868c804abad923eb3/tensorflow_padding/my_layers.py.

import tensorflow as tf
import tensorflow.keras as keras


class Padding2D(keras.layers.Layer):

    def __init__(self, pad_size=1, mode="symmetric", **kwargs):
        # assert mode == "symmetric" or mode == "reflect", \
        #    "Padding2D: mode has to be symmetric or reflect"
        self.pad_size = pad_size
        self.mode = mode
        super(Padding2D, self).__init__(**kwargs)

    def call(self, x):
        # rank = len(x.shape)
        # assert rank == 4, "Padding2D: tensor order must be 4"
        return tf.pad(x, [[0, 0], \
                          [self.pad_size, self.pad_size], \
                          [self.pad_size, self.pad_size], \
                          [0, 0]], \
                      mode=self.mode)

    # def build(self, input_shape):
    #     super(Padding2D, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        # rank = len(input_shape)
        # assert rank == 4, "Padding2D: tensor order must be 4"
        return (input_shape[0], \
                input_shape[1] + self.pad_size * 2, \
                input_shape[2] + self.pad_size * 2, \
                input_shape[3])
