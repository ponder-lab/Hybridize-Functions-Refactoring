import tensorflow as tf

from common import get_model


def test_call():
    get_model()(tf.ones([2, 4]))
