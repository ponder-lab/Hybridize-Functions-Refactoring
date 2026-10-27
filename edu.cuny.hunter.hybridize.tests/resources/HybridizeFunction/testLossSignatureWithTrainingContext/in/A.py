# From MusicTransformer-tensorflow2.0 (custom/callback.py, model.py, data.py, music-transformer-train.py), reduced.
import os
import pickle
import random
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow import keras


class TransformerLoss(keras.losses.SparseCategoricalCrossentropy):

    def __init__(self):
        super().__init__(from_logits=True, reduction="none")

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        loss = super().call(y_true, y_pred)
        return tf.reduce_mean(loss)


class MusicTransformer(keras.Model):

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = keras.layers.Embedding(vocab_size, 4)
        self.dense = keras.layers.Dense(vocab_size)

    def call(self, inputs):
        return self.dense(self.embedding(inputs))

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        reset_metrics=True,
        return_dict=False,
    ):
        return self.__train_step(x, y)

    def __train_step(self, inp, tar):
        with tf.GradientTape() as tape:
            predictions = self(inp)
            loss = self.loss(tar, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss


class Data:

    def __init__(self, files):
        self.files = files

    def batch(self, batch_size, length):
        batch_files = random.sample(self.files, k=batch_size)
        batch_data = [self._get_seq(file, length) for file in batch_files]
        return np.array(batch_data)

    def slide_seq2seq_batch(self, batch_size, length):
        data = self.batch(batch_size, length + 1)
        x = data[:, :-1]
        y = data[:, 1:]
        return x, y

    def _get_seq(self, fname, max_length):
        with open(fname, "rb") as f:
            data = pickle.load(f)
        start = random.randrange(0, len(data) - max_length)
        return data[start : start + max_length]


def preprocess(directory, count, vocab_size):
    files = []
    for i in range(count):
        fname = os.path.join(directory, str(i) + ".pickle")
        with open(fname, "wb") as f:
            pickle.dump([random.randrange(vocab_size) for _ in range(50)], f)
        files.append(fname)
    return files


def train(batch_size, max_seq, vocab_size):
    dataset = Data(preprocess(tempfile.mkdtemp(), 10, vocab_size))
    mt = MusicTransformer(vocab_size)
    mt.compile(optimizer=keras.optimizers.Adam(), loss=TransformerLoss())
    for _ in range(2):
        batch_x, batch_y = dataset.slide_seq2seq_batch(batch_size, max_seq)
        mt.train_on_batch(batch_x, batch_y)


train(8, 6, 2)

if __name__ == "__main__":
    loss = TransformerLoss()(
        np.array([[1], [0], [0]]),
        tf.constant([[0.5, 0.5], [0.1, 0.1], [0.1, 0.1]]),
    )
