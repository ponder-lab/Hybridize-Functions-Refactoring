# From https://github.com/ponder-lab/samples/blob/39f7644391e664244b45c90868c804abad923eb3/tensorflow_padding/tensorflow_padding.py.

#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from B import Padding2D


def plot_sample(imgs_g):
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(imgs_g[i])
        plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.show()


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype(np.float32) / 255.0

# Reflect
input_node = keras.layers.Input(shape=x_train.shape[1:])
pad = Padding2D(5, "reflect")(input_node)
