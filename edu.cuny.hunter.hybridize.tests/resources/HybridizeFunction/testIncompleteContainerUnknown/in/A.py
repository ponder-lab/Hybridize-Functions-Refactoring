# Fixture for a dict whose keys are read from a file: the analysis can't list them, and the
# values stored under them are ints, so nothing says `convert`'s `vocab` holds a tensor.
import tensorflow as tf


def load_vocab(path):
    vocab = {}
    with open(path) as reader:
        for line in reader:
            token = line.strip()
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def convert(vocab, items):
    return [vocab[item] for item in items]


vocab = load_vocab("vocab.txt")
print(convert(vocab, ["a", "b"]))
