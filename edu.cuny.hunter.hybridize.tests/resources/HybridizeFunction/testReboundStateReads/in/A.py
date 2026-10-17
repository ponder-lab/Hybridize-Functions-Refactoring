# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1076

import tensorflow as tf


# TensorFlowTTS's Tacotron2Sampler: setup_target rebinds the per-batch targets.
class Sampler:
    def __init__(self):
        self._ratio = tf.constant(1.0)

    def setup_target(self, targets, mel_lengths):
        self.mel_lengths = mel_lengths
        self.targets = targets
        self.max_lengths = mel_lengths + 0

    def next_inputs(self, time, outputs):
        finished = time + 1 >= self.max_lengths
        next_inputs = (
            self._ratio * self.targets[:, time, :] + (1.0 - self._ratio) * outputs
        )
        return finished, next_inputs


# TFTacotronLocationSensitiveAttention: setup_memory rebinds the keys and values, and setup_window the window.
class Attention:
    def __init__(self):
        self.v = tf.constant([[1.0], [1.0]])

    def setup_memory(self, memory):
        self.keys = memory
        self.values = memory

    def setup_window(self, win_front, win_back):
        self.win_front = win_front
        self.win_back = win_back
        self._indices = tf.tile(
            tf.expand_dims(tf.range(tf.shape(self.keys)[1]), 0),
            [tf.shape(self.keys)[0], 1],
        )

    def _compute_window_mask(self, max_alignments):
        expanded_max_alignments = tf.expand_dims(max_alignments, 1)
        low = expanded_max_alignments - self.win_front
        high = expanded_max_alignments + self.win_back
        mlow = tf.cast((self._indices < low), tf.float32)
        mhigh = tf.cast((self._indices > high), tf.float32)
        return mlow + mhigh

    def __call__(self, query, state):
        energy = tf.squeeze(tf.tanh(self.keys + tf.expand_dims(query, 1)) @ self.v, -1)
        alignments = tf.nn.softmax(energy)
        context = tf.reduce_sum(tf.expand_dims(alignments, 2) * self.values, 1)
        return context, alignments + state


# TFTacotronDecoder: step reads the sampler's and the attention's state through self.
class Decoder:
    def __init__(self):
        self.attention_layer = Attention()
        self.sampler = Sampler()

    def step(self, time, inputs, state):
        outputs, new_state = self.attention_layer(inputs, state)
        finished, next_inputs = self.sampler.next_inputs(time, outputs)
        return outputs, next_inputs


# OpenNMT-tf's AttentionalRNNDecoder: initialize rebinds the memory per batch.
class AttentionalRNNDecoder:
    def initialize(self, memory):
        self.memory = memory

    def _get_initial_state(self, initial_state):
        return tf.reduce_sum(self.memory, axis=1) + initial_state


# OpenNMT-tf's SequenceRecordInputter.make_features fills in the caller's features dictionary.
def make_features(element, features=None):
    if features is None:
        features = {}
    if "tensor" in features:
        return features
    features["length"] = tf.shape(element)[0]
    features["tensor"] = tf.cast(element, tf.float32)
    return features


# The same, called as OpenNMT-tf's Model calls it, with a copy of the caller's dictionary.
def make_features_of_copy(element, features=None):
    if features is None:
        features = {}
    if "tensor" in features:
        return features
    features["length"] = tf.shape(element)[0]
    features["tensor"] = tf.cast(element, tf.float32)
    return features


# Control: the field is set only in __init__.
class InitOnly:
    def __init__(self):
        self.scale = tf.constant(2.0)

    def scaled(self, x):
        return x * self.scale


# Control: the field is set only in build, which runs once.
class BuildOnly(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.offset = tf.ones(input_shape[-1:])

    def call(self, inputs):
        return inputs + self.offset


# Control: the field another method rebinds holds a string.
class StringField:
    def __init__(self):
        self.label = "a"

    def relabel(self, label):
        self.label = label

    def labeled(self, x):
        return tf.identity(x, name=self.label)


# Control: the field another method rebinds holds an int.
class IntField:
    def __init__(self):
        self.count = 1

    def recount(self, count):
        self.count = count

    def counted(self, x):
        return x * self.count


# Control: the field holds a variable that another method updates with assign.
class VariableField:
    def __init__(self):
        self.v = tf.Variable([1.0, 1.0])

    def update(self, value):
        self.v.assign(value)

    def read(self, x):
        return x + self.v


# The same, with the memory computed from a loop variable.
class LoopFedDecoder:
    def initialize(self, memory):
        self.memory = memory

    def _get_initial_state(self, initial_state):
        return tf.reduce_sum(self.memory, axis=1) + initial_state


# Control: the elements of a list argument, which tf.function passes in afresh on each call.
def sums_list(xs, y):
    return xs[0] + xs[1] + y


# The field is written by set, outside get.
class Holder:
    def set(self, value):
        self.value = value

    def get(self, x):
        return self.value + x


# Control: the holder is created, set, and read within one call.
def uses_fresh_holder(x):
    holder = Holder()
    holder.set(x)
    return holder.get(x)


# Another class's __init__ rebinds the memory's keys on every construction.
class Memory:
    def __init__(self):
        self.keys = tf.zeros([2])

    def read(self, x):
        return x + self.keys


class Frame:
    def __init__(self, memory, keys):
        memory.keys = keys


# A build outside Keras is an ordinary method, called per batch here.
class Builder:
    def build(self, batch):
        self.batch = batch

    def read(self, x):
        return x + self.batch


# Control: the field is set only in a base's __init__, which the subclass's __init__ calls.
class Base:
    def __init__(self):
        self.scale = tf.constant(3.0)


class Derived(Base):
    def __init__(self):
        super().__init__()

    def scaled(self, x):
        return x * self.scale


def memory(i):
    return tf.reshape(tf.range(8, dtype=tf.float32) * (i + 1.0), [1, 4, 2])


sampler = Sampler()
for i in range(2):
    sampler.setup_target(tf.fill([2, 3, 2], i + 1.0), tf.constant([3, 3]))
    sampler.next_inputs(tf.constant(0), tf.zeros([2, 2]))

attention = Attention()
attention.setup_memory(memory(0))
attention(tf.zeros([1, 2]), tf.zeros([1, 4]))
attention.setup_memory(memory(1))
attention(tf.zeros([1, 2]), tf.zeros([1, 4]))

window = Attention()
for i in range(2):
    window.setup_memory(tf.zeros([1, 4 + 2 * i, 2]))
    window.setup_window(tf.constant(1), tf.constant(1))
    window._compute_window_mask(tf.constant([1]))

decoder = Decoder()
decoder.attention_layer.setup_memory(memory(0))
decoder.sampler.setup_target(tf.fill([1, 3, 2], 1.0), tf.constant([3]))
decoder.step(tf.constant(0), tf.zeros([1, 2]), tf.zeros([1, 4]))
decoder.attention_layer.setup_memory(memory(1))
decoder.sampler.setup_target(tf.fill([1, 3, 2], 2.0), tf.constant([3]))
decoder.step(tf.constant(0), tf.zeros([1, 2]), tf.zeros([1, 4]))

rnn_decoder = AttentionalRNNDecoder()
rnn_decoder.initialize(memory(0))
rnn_decoder._get_initial_state(tf.zeros([1, 2]))
rnn_decoder.initialize(memory(1))
rnn_decoder._get_initial_state(tf.zeros([1, 2]))

loop_fed_decoder = LoopFedDecoder()
for i in range(2):
    loop_fed_decoder.initialize(memory(i))
    loop_fed_decoder._get_initial_state(tf.zeros([1, 2]))

caller = {"other": tf.constant(0)}
make_features(tf.constant([1, 2, 3]), caller)

copied = {"other": tf.constant(0)}
make_features_of_copy(tf.constant([1, 2, 3]), features=copied.copy())

InitOnly().scaled(tf.constant([1.0, 2.0]))

build_only = BuildOnly()
build_only(tf.constant([[1.0, 2.0]]))
build_only(tf.constant([[3.0, 4.0]]))

string_field = StringField()
string_field.labeled(tf.constant([1.0, 2.0]))
string_field.relabel("b")
string_field.labeled(tf.constant([1.0, 2.0]))

int_field = IntField()
int_field.counted(tf.constant([1.0, 2.0]))
int_field.recount(2)
int_field.counted(tf.constant([1.0, 2.0]))

variable_field = VariableField()
variable_field.read(tf.constant([1.0, 2.0]))
variable_field.update(tf.constant([2.0, 2.0]))
variable_field.read(tf.constant([1.0, 2.0]))

sums_list([tf.constant(1.0), tf.constant(2.0)], tf.constant(0.0))
sums_list([tf.constant(3.0), tf.constant(4.0)], tf.constant(0.0))

uses_fresh_holder(tf.constant([1.0, 2.0]))
uses_fresh_holder(tf.constant([3.0, 4.0]))

shared_memory = Memory()
for i in range(2):
    Frame(shared_memory, tf.fill([2], i + 1.0))
    shared_memory.read(tf.constant([1.0, 2.0]))

builder = Builder()
for i in range(2):
    builder.build(tf.fill([2], i + 1.0))
    builder.read(tf.constant([1.0, 2.0]))

derived = Derived()
derived.scaled(tf.constant([1.0, 2.0]))
derived.scaled(tf.constant([3.0, 4.0]))
