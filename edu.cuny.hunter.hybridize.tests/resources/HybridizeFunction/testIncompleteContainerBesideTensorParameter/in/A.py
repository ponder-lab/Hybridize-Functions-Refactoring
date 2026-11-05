# Fixture for a function with a tensor parameter beside one receiving a dict whose keys are
# read from a file: the analysis can't list them, so whether `vocab` holds a tensor is unknown.
import tensorflow as tf


def load_vocab(path):
    vocab = {}
    with open(path) as reader:
        for line in reader:
            vocab[line.strip()] = len(vocab)
    return vocab


def scale(x, vocab):
    return x * len(vocab)


scale(tf.ones([2]), load_vocab("vocab.txt"))
