# Vendored from MusicTransformer-tensorflow2.0's custom/layers.py: the encoder/decoder layer
# chain the two model classes instantiate, verbatim except for dropped unused classes.
import math

import math as m
import numpy as np
import tensorflow as tf
from tensorflow import keras


class DynamicPositionEmbedding(keras.layers.Layer):
    def __init__(self, embedding_dim, max_seq=2048, **kwargs):
        super().__init__(**kwargs)
        embed_sinusoid_list = np.array(
            [
                [
                    [
                        m.sin(
                            pos
                            * m.exp(-m.log(10000) * i / embedding_dim)
                            * m.exp(m.log(10000) / embedding_dim * (i % 2))
                            + 0.5 * m.pi * (i % 2)
                        )
                        for i in range(embedding_dim)
                    ]
                    for pos in range(max_seq)
                ]
            ]
        )
        self.positional_embedding = tf.constant(embed_sinusoid_list, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        return tf.add(inputs, self.positional_embedding[:, : inputs.shape[1], :])


class RelativeGlobalAttention(keras.layers.Layer):
    def __init__(self, h=4, d=256, add_emb=False, max_seq=2048, **kwargs):
        super().__init__(**kwargs)
        self.len_k = None
        self.max_seq = max_seq
        self.E = None
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = keras.layers.Dense(int(self.d))
        self.Wk = keras.layers.Dense(int(self.d))
        self.Wv = keras.layers.Dense(int(self.d))
        self.fc = keras.layers.Dense(d)
        self.additional = add_emb
        if self.additional:
            self.Radd = None

    def build(self, input_shape):
        self.shape_q = input_shape[0][1]
        self.shape_k = input_shape[1][1]
        self.E = self.add_weight("emb", shape=[self.max_seq, int(self.dh)])

    def call(self, inputs, mask=None, **kwargs):
        q = inputs[0]
        q = self.Wq(q)
        q = tf.reshape(q, (q.shape[0], q.shape[1], self.h, -1))
        q = tf.transpose(q, (0, 2, 1, 3))  # batch, h, seq, dh

        k = inputs[1]
        k = self.Wk(k)
        k = tf.reshape(k, (k.shape[0], k.shape[1], self.h, -1))
        k = tf.transpose(k, (0, 2, 1, 3))

        v = inputs[2]
        v = self.Wv(v)
        v = tf.reshape(v, (v.shape[0], v.shape[1], self.h, -1))
        v = tf.transpose(v, (0, 2, 1, 3))

        self.len_k = k.shape[2]
        self.len_q = q.shape[2]

        E = self._get_left_embedding(self.len_q, self.len_k)
        QE = tf.einsum("bhld,md->bhlm", q, E)
        QE = self._qe_masking(QE)
        Srel = self._skewing(QE)

        Kt = tf.transpose(k, [0, 1, 3, 2])
        QKt = tf.matmul(q, Kt)
        logits = QKt + Srel
        logits = logits / math.sqrt(self.dh)

        if mask is not None:
            logits += tf.cast(mask, tf.float32) * -1e9

        attention_weights = tf.nn.softmax(logits, -1)
        attention = tf.matmul(attention_weights, v)

        out = tf.transpose(attention, (0, 2, 1, 3))
        out = tf.reshape(out, (out.shape[0], -1, self.d))

        out = self.fc(out)
        return out, attention_weights

    def _get_left_embedding(self, len_q, len_k):
        starting_point = max(0, self.max_seq - len_q)
        e = self.E[starting_point:, :]
        return e

    @staticmethod
    def _qe_masking(qe):
        mask = tf.sequence_mask(
            tf.range(qe.shape[-1] - 1, qe.shape[-1] - qe.shape[-2] - 1, -1),
            qe.shape[-1],
        )

        mask = tf.logical_not(mask)
        mask = tf.cast(mask, tf.float32)

        return mask * qe

    def _skewing(self, tensor: tf.Tensor):
        padded = tf.pad(tensor, [[0, 0], [0, 0], [0, 0], [1, 0]])
        reshaped = tf.reshape(
            padded, shape=[-1, padded.shape[1], padded.shape[-1], padded.shape[-2]]
        )
        Srel = reshaped[:, :, 1:, :]

        if self.len_k > self.len_q:
            Srel = tf.pad(Srel, [[0, 0], [0, 0], [0, 0], [0, self.len_k - self.len_q]])
        elif self.len_k < self.len_q:
            Srel = Srel[:, :, :, : self.len_k]

        return Srel


class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, rate=0.1, h=16, additional=False, max_seq=2048):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.rga = RelativeGlobalAttention(
            h=h, d=d_model, max_seq=max_seq, add_emb=additional
        )

        self.FFN_pre = keras.layers.Dense(self.d_model // 2, activation=tf.nn.relu)
        self.FFN_suf = keras.layers.Dense(self.d_model)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, x, mask=None, training=False, **kwargs):
        attn_out, w = self.rga([x, x, x], mask)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.layernorm1(attn_out + x)

        ffn_out = self.FFN_pre(out1)
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout2(ffn_out, training=training)
        out2 = self.layernorm2(out1 + ffn_out)
        return out2, w


class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_model, rate=0.1, h=16, additional=False, max_seq=2048):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.rga2 = RelativeGlobalAttention(
            d=d_model, h=h, max_seq=max_seq, add_emb=additional
        )
        self.rga = RelativeGlobalAttention(
            d=d_model, h=h, max_seq=max_seq, add_emb=additional
        )

        self.FFN_pre = keras.layers.Dense(self.d_model // 2, activation=tf.nn.relu)
        self.FFN_suf = keras.layers.Dense(self.d_model)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)

    def call(
        self,
        x,
        enc_output,
        mask=None,
        lookup_mask=None,
        training=False,
        w_out=False,
        **kwargs
    ):
        attn_out, aw1 = self.rga([x, x, x], mask=lookup_mask)
        attn_out = self.dropout1(attn_out, training=training)
        attn_out = self.layernorm1(attn_out + x)

        attn_out2, aw2 = self.rga2([attn_out, enc_output, enc_output], mask=mask)
        attn_out2 = self.dropout2(attn_out2, training=training)
        attn_out2 = self.layernorm2(attn_out2 + attn_out)

        ffn_out = self.FFN_pre(attn_out2)
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout3(ffn_out, training=training)
        out = self.layernorm3(attn_out2 + ffn_out)

        if w_out:
            return out, aw1, aw2
        else:
            return out


class Encoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, input_vocab_size, rate=0.1, max_len=None):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = keras.layers.Embedding(input_vocab_size, d_model)
        if True:
            self.pos_encoding = DynamicPositionEmbedding(self.d_model, max_seq=max_len)

        self.enc_layers = [
            EncoderLayer(
                d_model, rate, h=self.d_model // 64, additional=False, max_seq=max_len
            )
            for i in range(num_layers)
        ]
        self.dropout = keras.layers.Dropout(rate)

    def call(self, x, mask=None, training=False):
        weights = []
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, w = self.enc_layers[i](x, mask, training=training)
            weights.append(w)
        return x, weights  # (batch_size, input_seq_len, d_model)


class Decoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, input_vocab_size, rate=0.1, max_len=None):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(input_vocab_size, d_model)

        if True:
            self.pos_encoding = DynamicPositionEmbedding(self.d_model, max_seq=max_len)

        self.dec_layers = [
            DecoderLayer(
                d_model, rate, h=self.d_model // 64, additional=False, max_seq=max_len
            )
            for i in range(num_layers)
        ]
        self.dropout = keras.layers.Dropout(rate)

    def call(self, x, mask, lookup_mask, training, enc_output=None):
        weights = []
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, w1, w2 = self.dec_layers[i](
                x,
                enc_output,
                lookup_mask=lookup_mask,
                mask=mask,
                training=training,
                w_out=True,
            )
            weights.append((w1, w2))
        return x, weights
