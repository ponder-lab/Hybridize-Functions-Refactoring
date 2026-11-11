import os
import tempfile

import tensorflow as tf

model = tf.keras.Sequential([tf.keras.layers.Dense(4)])
model(tf.ones([2, 4]))
model.save_weights(os.path.join(tempfile.mkdtemp(), "t.h5"))
