# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1165

# An index that a holder creates and fills when it is handed a dataset, and that a caller may also create and fill itself before
# handing it to the holder.

import tensorflow as tf


# The index's search puts the fields it reads in a tuple.
class Stream(tf.keras.Model):
    def __init__(self, k=10):
        super().__init__()
        self._k = k
        self._candidates = None
        self._identifiers = None

    def index(self, candidates, identifiers=None, **kwargs):
        self._candidates = candidates
        self._identifiers = identifiers
        return self

    def call(self, queries, k=None, **kwargs):
        if self._identifiers is not None:
            dataset = tf.data.Dataset.zip((self._identifiers, self._candidates))
        else:
            dataset = self._candidates.enumerate()

        result = tf.zeros([])

        for _, batch in dataset:
            result += tf.reduce_sum(tf.matmul(queries, batch, transpose_b=True))

        return result


class Brute(tf.keras.Model):
    def __init__(self, k=10):
        super().__init__()
        self._k = k

    def index(self, candidates, identifiers=None):
        self._candidates = tf.concat(list(candidates), axis=0)
        return self

    def call(self, queries, k=None, **kwargs):
        return tf.reduce_sum(tf.matmul(queries, self._candidates, transpose_b=True))


class Metric(tf.keras.layers.Layer):
    def __init__(self, candidates, k=100):
        super().__init__()

        if isinstance(candidates, tf.data.Dataset):
            candidates = Stream(k=k).index(candidates)

        self._candidates = candidates
        self._k = k

    def update(self, queries):
        return self._candidates(queries, k=self._k)


def measure(top_k_layer, queries):
    candidates = tf.data.Dataset.from_tensor_slices(tf.ones([4, 2])).batch(2)

    if top_k_layer is not None:
        candidates = top_k_layer().index(candidates)

    metric = Metric(candidates=candidates)
    return metric.update(queries)


# The same, with a search that puts the field it reads in no tuple.
class Lone(tf.keras.Model):
    def __init__(self, k=10):
        super().__init__()
        self._k = k
        self._candidates = None
        self._identifiers = None

    def index(self, candidates, identifiers=None, **kwargs):
        self._candidates = candidates
        self._identifiers = identifiers
        return self

    def call(self, queries, k=None, **kwargs):
        result = tf.zeros([])

        for batch in self._candidates:
            result += tf.reduce_sum(tf.matmul(queries, batch, transpose_b=True))

        return result


class Gauge(tf.keras.layers.Layer):
    def __init__(self, candidates, k=100):
        super().__init__()

        if isinstance(candidates, tf.data.Dataset):
            candidates = Lone(k=k).index(candidates)

        self._candidates = candidates
        self._k = k

    def update(self, queries):
        return self._candidates(queries, k=self._k)


def gauge(top_k_layer, queries):
    candidates = tf.data.Dataset.from_tensor_slices(tf.ones([4, 2])).batch(2)

    if top_k_layer is not None:
        candidates = top_k_layer().index(candidates)

    metric = Gauge(candidates=candidates)
    return metric.update(queries)


# The same, with a search that puts the fields it reads in a list.
class Boxed(tf.keras.Model):
    def __init__(self, k=10):
        super().__init__()
        self._k = k
        self._candidates = None
        self._identifiers = None

    def index(self, candidates, identifiers=None, **kwargs):
        self._candidates = candidates
        self._identifiers = identifiers
        return self

    def call(self, queries, k=None, **kwargs):
        boxes = [self._identifiers, self._candidates]
        result = tf.zeros([])

        for batch in boxes[1]:
            result += tf.reduce_sum(tf.matmul(queries, batch, transpose_b=True))

        return result


class Crate(tf.keras.layers.Layer):
    def __init__(self, candidates, k=100):
        super().__init__()

        if isinstance(candidates, tf.data.Dataset):
            candidates = Boxed(k=k).index(candidates)

        self._candidates = candidates
        self._k = k

    def update(self, queries):
        return self._candidates(queries, k=self._k)


def crate(top_k_layer, queries):
    candidates = tf.data.Dataset.from_tensor_slices(tf.ones([4, 2])).batch(2)

    if top_k_layer is not None:
        candidates = top_k_layer().index(candidates)

    metric = Crate(candidates=candidates)
    return metric.update(queries)


print(measure(Stream, tf.ones([3, 2])))
print(measure(Brute, tf.ones([3, 2])))
print(measure(None, tf.ones([3, 2])))
print(gauge(Lone, tf.ones([3, 2])))
print(gauge(None, tf.ones([3, 2])))
print(crate(Boxed, tf.ones([3, 2])))
print(crate(None, tf.ones([3, 2])))
