import tensorflow as tf

# Reduced from NLPGNN's `crf_decode_forward`. A Keras `RNN` called with an explicit initial state hands its cell that state wrapped in
# a list, and the cell reads `state[0]`.


class Cell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, units, **kwargs):
        super(Cell, self).__init__(**kwargs)
        self.units = units

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def call(self, inputs, state):
        output = state[0] + inputs
        return output, [output]


def decode_forward(inputs, state):
    layer = tf.keras.layers.RNN(Cell(3), return_sequences=True, return_state=True)
    return layer(inputs, state)


decode_forward(tf.ones([2, 5, 3]), tf.ones([2, 3]))
