# Reduced from transformer, transformer/transformer.py and transformer/text_classification.py. The embedding's `output_dim` is
# a value the analysis can't read; the subject reads it from a command-line flag.
import os

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input, Layer


class MultiplyConstant(Layer):
    def __init__(self, c, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def call(self, inputs):
        return inputs * self.c

    def get_config(self):
        base = super().get_config()
        return dict(list(base.items()) + [("c", self.c)])


class Encoder:
    def __init__(self, d_model, vocab_size, scope="encoder"):
        self.d_model = d_model
        self.scope = scope
        self.embedding = Embedding(
            input_dim=vocab_size, output_dim=d_model, name="%s/embedding" % scope
        )

    def __call__(self, x):
        x = self.embedding(x)
        return MultiplyConstant(self.d_model, name="%s/multiply" % self.scope)(x)


inp = Input((None,), dtype="int32", name="inp")
net = Encoder(d_model=int(os.environ.get("D_MODEL", "128")), vocab_size=100)(inp)
assert net.shape.as_list() == [None, None, 128] and net.dtype == tf.float32
