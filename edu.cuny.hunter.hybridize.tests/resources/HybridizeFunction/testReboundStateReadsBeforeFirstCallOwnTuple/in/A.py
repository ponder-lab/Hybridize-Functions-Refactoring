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


# A helper creates the tuple, for the function replacing an index and for the search alike. The search reads the index from a
# tuple that is replaced between two searches.
def pair_of(first, second):
    return (first, second)


class Helped:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


class Helper:
    def __init__(self):
        self.pair = None

    def replace(self, candidates):
        index = Helped()
        index.index(candidates)
        self.pair = pair_of(index, candidates)

    def search(self, queries):
        both = pair_of(queries, queries)
        return self.pair[0].search(both[0])


# The same, with a helper that creates the tuple through another, so that one function creates both tuples whichever function asks.
def pair_in(first, second):
    return (first, second)


def pair_through(first, second):
    return pair_in(first, second)


class Relayed:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


class Relayer:
    def __init__(self):
        self.pair = None

    def replace(self, candidates):
        index = Relayed()
        index.index(candidates)
        self.pair = pair_through(index, candidates)

    def search(self, queries):
        both = pair_through(queries, queries)
        return self.pair[0].search(both[0])


# The same, with the object holding the tuple passed to plain functions.
class Routed:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


class Route:
    def __init__(self):
        self.pair = None


def replace_routed(route, candidates):
    index = Routed()
    index.index(candidates)
    route.pair = pair_through(index, candidates)


def search_routed(route, queries):
    both = pair_through(queries, queries)
    return route.pair[0].search(both[0])


# The same, with the tuple in a module's variable.
PAIR = None


def lend(candidates):
    global PAIR
    index = Helped()
    index.index(candidates)
    PAIR = pair_of(index, candidates)


def search_lent(queries):
    both = pair_of(queries, queries)
    return PAIR[0].search(both[0])


# A tuple that the caller creates and passes on. A trace is keyed on what a tuple argument holds, so each index has its own.
class Passed:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_passed(pair, queries):
    return pair[0].search(queries)


def run_passed(candidates, queries):
    index = Passed()
    index.index(candidates)
    return search_passed((index, candidates), queries)


print(measure(Stream, tf.ones([3, 2])))
print(measure(Brute, tf.ones([3, 2])))
print(measure(None, tf.ones([3, 2])))
print(gauge(Lone, tf.ones([3, 2])))
print(gauge(None, tf.ones([3, 2])))
print(crate(Boxed, tf.ones([3, 2])))
print(crate(None, tf.ones([3, 2])))

helper = Helper()
for i in range(2):
    helper.replace(tf.fill([3, 2], i + 1.0))
    print(helper.search(tf.ones([1, 2])))

relayer = Relayer()
for i in range(2):
    relayer.replace(tf.fill([3, 2], i + 1.0))
    print(relayer.search(tf.ones([1, 2])))

route = Route()
for i in range(2):
    replace_routed(route, tf.fill([3, 2], i + 1.0))
    print(search_routed(route, tf.ones([1, 2])))

for i in range(2):
    lend(tf.fill([3, 2], i + 1.0))
    print(search_lent(tf.ones([1, 2])))

for i in range(2):
    print(run_passed(tf.fill([3, 2], i + 1.0), tf.ones([1, 2])))
