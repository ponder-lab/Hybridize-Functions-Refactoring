# Vendored from MusicTransformer-tensorflow2.0's model.py: both model classes reduced to the
# structure the keyword-collision pin needs. `MusicTransformerDecoder.sanity_check(self, x, y,
# mode='v', step=None)` is the corrected function of wala/ML#740: it is reached from two call
# sites with the same total argument count but different keyword-name sets (`step` from the
# training script, `mode` from `train_on_batch` below). The sibling `MusicTransformer` class
# with its same-named methods is retained deliberately; the collision did not reproduce in a
# minimal single-class reduction (see #791).
from custom.layers import *
import params as par
import sys
from tensorflow.python import keras
import utils

tf.executing_eagerly()


class MusicTransformer(keras.Model):
    def __init__(
        self,
        embedding_dim=64,
        vocab_size=par.vocab_size,
        num_layer=1,
        max_seq=8,
        dropout=0.2,
        debug=False,
        dist=False,
    ):
        super(MusicTransformer, self).__init__()
        self._debug = debug
        self.max_seq = max_seq
        self.num_layer = num_layer
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dist = dist

        self.Encoder = Encoder(
            num_layers=self.num_layer,
            d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size,
            rate=dropout,
            max_len=max_seq,
        )
        self.Decoder = Decoder(
            num_layers=self.num_layer,
            d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size,
            rate=dropout,
            max_len=max_seq,
        )
        self.fc = keras.layers.Dense(self.vocab_size, activation=None, name="output")

        self._set_metrics()

    def call(
        self,
        inputs,
        targets,
        training=None,
        eval=None,
        src_mask=None,
        trg_mask=None,
        lookup_mask=None,
    ):
        encoder, weight_encoder = self.Encoder(inputs, training=training, mask=src_mask)
        decoder, weights = self.Decoder(
            targets,
            enc_output=encoder,
            training=training,
            lookup_mask=lookup_mask,
            mask=trg_mask,
        )

        fc = self.fc(decoder)
        if training:
            return fc
        elif eval:
            return fc, weights
        else:
            return tf.nn.softmax(fc)

    def train_on_batch(
        self, x, y=None, sample_weight=None, class_weight=None, reset_metrics=True
    ):
        if self._debug:
            tf.print(
                "sanity:\n", self.sanity_check(x, y, mode="d"), output_stream=sys.stdout
            )

        x, dec_input, target = self.__prepare_train_data(x, y)

        enc_mask, tar_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(
            self.max_seq, x, dec_input
        )

        predictions = self.__train_step(
            x, dec_input, target, enc_mask, tar_mask, look_ahead_mask, True
        )

        result_metric = []
        loss = tf.reduce_mean(self.loss_value)
        loss = tf.reduce_mean(loss)
        for metric in self.custom_metrics:
            result_metric.append(metric(target, predictions).numpy())

        return [loss.numpy()] + result_metric

    def __train_step(
        self, inp, inp_tar, out_tar, enc_mask, tar_mask, lookup_mask, training
    ):
        with tf.GradientTape() as tape:
            predictions = self.call(
                inp,
                targets=inp_tar,
                src_mask=enc_mask,
                trg_mask=tar_mask,
                lookup_mask=lookup_mask,
                training=training,
            )
            self.loss_value = self.loss(out_tar, predictions)
        gradients = tape.gradient(self.loss_value, self.trainable_variables)
        self.grad = gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return predictions

    def sanity_check(self, x, y, mode="v"):
        # mode: v -> vector, d -> dict
        x, inp_tar, out_tar = MusicTransformer.__prepare_train_data(x, y)

        enc_mask, tar_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(
            self.max_seq, x, inp_tar
        )
        predictions = self.call(
            x,
            targets=inp_tar,
            src_mask=enc_mask,
            trg_mask=tar_mask,
            lookup_mask=look_ahead_mask,
            training=False,
        )

        if mode == "v":
            return predictions
        elif mode == "d":
            dic = {}
            for row in tf.argmax(predictions, -1).numpy():
                for col in row:
                    try:
                        dic[str(col)] += 1
                    except KeyError:
                        dic[str(col)] = 1
            return dic
        else:
            return tf.argmax(predictions, -1)

    @staticmethod
    def __prepare_train_data(x, y):
        start_token = tf.ones((y.shape[0], 1), dtype=y.dtype) * par.token_sos

        out_tar = y
        inp_tar = y[:, :-1]
        inp_tar = tf.concat([start_token, inp_tar], -1)
        return x, inp_tar, out_tar

    def _set_metrics(self):
        self.custom_metrics = [keras.metrics.SparseCategoricalAccuracy()]

    def reset_metrics(self):
        for metric in self.custom_metrics:
            metric.reset_states()
        return


class MusicTransformerDecoder(keras.Model):
    def __init__(
        self,
        embedding_dim=64,
        vocab_size=par.vocab_size,
        num_layer=1,
        max_seq=8,
        dropout=0.2,
        debug=False,
        dist=False,
    ):
        super(MusicTransformerDecoder, self).__init__()

        self._debug = debug
        self.max_seq = max_seq
        self.num_layer = num_layer
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dist = dist

        self.Decoder = Encoder(
            num_layers=self.num_layer,
            d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size,
            rate=dropout,
            max_len=max_seq,
        )
        self.fc = keras.layers.Dense(self.vocab_size, activation=None, name="output")

        self._set_metrics()

    def call(self, inputs, training=None, eval=None, lookup_mask=None):
        decoder, w = self.Decoder(inputs, training=training, mask=lookup_mask)
        fc = self.fc(decoder)
        if training:
            return fc
        elif eval:
            return fc, w
        else:
            return tf.nn.softmax(fc)

    def train_on_batch(
        self, x, y=None, sample_weight=None, class_weight=None, reset_metrics=True
    ):
        if self._debug:
            tf.print(
                "sanity:\n", self.sanity_check(x, y, mode="d"), output_stream=sys.stdout
            )

        x, y = self.__prepare_train_data(x, y)

        _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, x)

        predictions = self.__train_step(x, y, look_ahead_mask, True)

        result_metric = []
        loss = tf.reduce_mean(self.loss_value)
        loss = tf.reduce_mean(loss)
        for metric in self.custom_metrics:
            result_metric.append(metric(y, predictions).numpy())

        return [loss.numpy()] + result_metric

    def __train_step(self, inp_tar, out_tar, lookup_mask, training):
        with tf.GradientTape() as tape:
            predictions = self.call(
                inputs=inp_tar, lookup_mask=lookup_mask, training=training
            )
            self.loss_value = self.loss(out_tar, predictions)
        gradients = tape.gradient(self.loss_value, self.trainable_variables)
        self.grad = gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return predictions

    def sanity_check(self, x, y, mode="v", step=None):
        # mode: v -> vector, d -> dict
        _, tar_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(
            self.max_seq, x, x
        )
        predictions = self.call(x, lookup_mask=look_ahead_mask, training=False)

        if mode == "v":
            tf.summary.image("vector", tf.expand_dims(predictions, -1), step)
            return predictions
        elif mode == "d":
            dic = {}
            for row in tf.argmax(predictions, -1).numpy():
                for col in row:
                    try:
                        dic[str(col)] += 1
                    except KeyError:
                        dic[str(col)] = 1
            return dic
        else:
            tf.summary.image("tokens", tf.argmax(predictions, -1), step)
            return tf.argmax(predictions, -1)

    @staticmethod
    def __prepare_train_data(x, y):
        # method without eos
        return x, y

    def _set_metrics(self):
        self.custom_metrics = [keras.metrics.SparseCategoricalAccuracy()]

    def reset_metrics(self):
        for metric in self.custom_metrics:
            metric.reset_states()
        return


# Vendored/reduced from MusicTransformer-tensorflow2.0's train.py: the training-script call
# site `mt.sanity_check(eval_x, eval_y, step=e)` that collides with `train_on_batch`'s
# `self.sanity_check(x, y, mode='d')` on total argument count while differing in keyword
# names (wala/ML#740). Both models run with debug enabled so every vendored call site
# executes eagerly.
max_seq = 8

mt = MusicTransformerDecoder(
    embedding_dim=64,
    vocab_size=par.vocab_size,
    num_layer=1,
    max_seq=max_seq,
    dropout=0.2,
    debug=True,
)
mt.compile(
    optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

batch_x = tf.ones((2, max_seq), dtype=tf.int32)
batch_y = tf.ones((2, max_seq), dtype=tf.int32)
eval_x = tf.ones((2, max_seq), dtype=tf.int32)
eval_y = tf.ones((2, max_seq), dtype=tf.int32)

for e in range(2):
    for b in range(1):
        result_metrics = mt.train_on_batch(batch_x, batch_y)
        if b == 0:
            mt.sanity_check(eval_x, eval_y, step=e)

assert len(result_metrics) == 2

mt2 = MusicTransformer(
    embedding_dim=64,
    vocab_size=par.vocab_size,
    num_layer=1,
    max_seq=max_seq,
    dropout=0.2,
    debug=True,
)
mt2.compile(
    optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

assert len(mt2.train_on_batch(batch_x, batch_y)) == 2
