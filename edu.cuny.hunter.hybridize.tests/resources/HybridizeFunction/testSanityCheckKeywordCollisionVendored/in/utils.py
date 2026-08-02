# Vendored from MusicTransformer-tensorflow2.0's utils.py: the one helper the sanity-check
# slice reaches, verbatim.
import tensorflow as tf

import params as par


def get_masked_with_pad_tensor(size, src, trg):
    """
    :param size: the size of target input
    :param src: source tensor
    :param trg: target tensor
    :return:
    """
    src = tf.cast(src[:, tf.newaxis, tf.newaxis, :], tf.int32)
    trg = tf.cast(trg[:, tf.newaxis, tf.newaxis, :], tf.int32)
    src_pad_tensor = tf.ones_like(src) * par.pad_token
    src_mask = tf.cast(tf.equal(src, src_pad_tensor), dtype=tf.int32)
    trg_mask = tf.cast(tf.equal(src, src_pad_tensor), dtype=tf.int32)
    if trg is not None:
        trg_pad_tensor = tf.ones_like(trg) * par.pad_token
        dec_trg_mask = tf.cast(tf.equal(trg, trg_pad_tensor), dtype=tf.int32)
        # boolean reversing i.e) True * -1 + 1 = False
        seq_mask = (
            tf.sequence_mask(list(range(1, size + 1)), size, dtype=tf.int32) * -1 + 1
        )
        look_ahead_mask = tf.cast(tf.maximum(dec_trg_mask, seq_mask), dtype=tf.int32)
    else:
        trg_mask = None
        look_ahead_mask = None

    return src_mask, trg_mask, look_ahead_mask
