# Fixture for #1169 (distilling TensorSpeech-TensorFlowASR's
# tests/losses/test_rnnt_loss.py `WarpRNNTTest._run_rnnt`): passing a tensor
# parameter's value to `tf.constant` works eagerly, where the argument is a NumPy
# array or an eager tensor, but raises under tracing, where it is symbolic
# (TypeError: Expected any non-tensor type, but got a tensor instead), so the
# conversion must be declined. Converting the parameter with
# `tf.convert_to_tensor`, passing `tf.constant` the static dimensions of a
# parameter, its rank, or a NumPy array built from them, which tracing keeps
# concrete, and passing it a literal remain convertible.
import numpy as np
import tensorflow as tf


class WarpRNNTTest:
    def _run_rnnt(self, acts, labels):
        acts_t = tf.constant(acts)
        labels_t = tf.constant(labels)
        return tf.reduce_sum(acts_t) + tf.cast(tf.reduce_sum(labels_t), tf.float32)


def constant_param(a):
    return tf.constant(a) * 2


def constant_derived(a):
    return tf.constant(a + 1.0)


def constant_keyword(a):
    return tf.constant(value=a, dtype=tf.float32)


def convert_param(a):
    return tf.convert_to_tensor(a) * 2


def constant_shape(a):
    return tf.constant(list(a.shape))


def constant_literal(a):
    return tf.constant(2.0) * a


def constant_rank(a):
    return tf.constant(len(a.shape))


def constant_arange(a):
    return tf.constant(np.arange(a.shape[0]), tf.float32) + a


acts = np.array([0.1, 0.7, 0.2], dtype=np.float32)
labels = np.array([1, 2], dtype=np.int32)

assert abs(float(WarpRNNTTest()._run_rnnt(acts, labels)) - 4.0) < 1e-6
assert list(constant_param(acts).numpy()) == list(acts * 2)
assert list(constant_derived(acts).numpy()) == list(acts + 1.0)
assert list(constant_keyword(acts).numpy()) == list(acts)
assert list(convert_param(acts).numpy()) == list(acts * 2)
assert list(constant_shape(acts).numpy()) == [3]
assert list(constant_literal(acts).numpy()) == list(acts * 2)
assert int(constant_rank(acts)) == 1
assert np.allclose(constant_arange(acts).numpy(), np.arange(3) + acts)
