import tensorflow as tf


class MLP(tf.keras.layers.Layer):
    def build(self, input_shape):
        hidden_size = input_shape[-1]
        self.c_fc = tf.keras.layers.Dense(hidden_size * 4, name="c_fc")
        self.c_proj = tf.keras.layers.Dense(hidden_size, name="c_proj")

    def call(self, input, training=True):
        h = self.c_fc(input)
        h2 = self.c_proj(h)
        return h2


class Block(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.mlp = MLP(name="mlp")

    def call(self, input, past=None, training=False):
        mlp_output = self.mlp(input, training=training)
        residual_output = mlp_output + input
        return residual_output, past


class Model(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.encoder_layers = []
        for layer_idx in range(2):
            self.encoder_layers.append(Block(name="h{}".format(layer_idx)))

    def call(self, hidden_states, past=None, training=False):
        presents = []
        pasts = [None, None]
        for i, (block, layer_past) in enumerate(zip(self.encoder_layers, pasts)):
            hidden_states, present = block(hidden_states, layer_past, training)
            presents.append(present)
        return hidden_states


# Reduces NLPGNN's GPT2 encoder loop (#872): the blocks are reached only through the
# enumerate-loop pair binding (`for i, (block, layer_past) in enumerate(zip(...))`),
# which binds real elements only since Ariadne 0.52.82 made `enumerate` yield
# (index, element) tuples (wala/ML#826).
model = Model()
x = tf.ones([1, 2, 4])
y = model(x)
