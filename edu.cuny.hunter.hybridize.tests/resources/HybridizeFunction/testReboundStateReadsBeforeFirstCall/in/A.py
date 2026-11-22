# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1165

# Fields written only on an object that the writing function has just created, before anything reading them runs. Each object's field
# holds one value from its first read on, so a traced read of it is never stale.

import tensorflow as tf


# The function creating the trainer calls the method creating its optimizer, and stores two more fields, before training starts.
class Trainer(tf.keras.Model):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.metric = None
        self.weight = tf.Variable(1.0)

    def create_optimizer(self):
        with tf.name_scope("optimizer"):
            if self.learning_rate is None:
                self.optimizer = tf.keras.optimizers.Adam()
            else:
                self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        return self.optimizer

    def step(self, x):
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(self.weight * x) * self.scale
        gradients = tape.gradient(loss, [self.weight])
        self.optimizer.apply_gradients(zip(gradients, [self.weight]))
        if self.metric is not None:
            self.metric.update_state(loss)
        return loss

    def fit(self, x):
        for _ in range(2):
            print(self.step(x))


def train(scaled):
    if scaled:
        with tf.name_scope("scaled"):
            trainer = Trainer(0.5)
            trainer.create_optimizer()
        trainer.metric = tf.keras.metrics.Mean()
        trainer.scale = tf.constant(2.0)
    else:
        trainer = Trainer(None)
        trainer.create_optimizer()
        trainer.scale = tf.constant(1.0)
    trainer.fit(tf.constant([1.0, 2.0]))


# Another layer's constructor creates the retrieval model and indexes the candidates on it in one expression.
class Streaming(tf.keras.Model):
    def __init__(self, k):
        super().__init__()
        self._k = k
        self._candidates = None
        self._identifiers = None

    def index(self, candidates, identifiers=None):
        self._candidates = candidates
        self._identifiers = identifiers
        return self

    def call(self, queries):
        if self._candidates is None:
            raise ValueError("The `index` method must be called first.")

        def top_scores(candidates):
            scores = tf.matmul(queries, candidates, transpose_b=True)
            return tf.math.top_k(scores, k=self._k).values

        def best(state, scores):
            return tf.maximum(state, scores)

        initial = tf.zeros([tf.shape(queries)[0], self._k])
        return (
            self._candidates.map(top_scores).reduce(initial, best) + self._identifiers
        )


class Retrieval(tf.keras.layers.Layer):
    def __init__(self, candidates, identifiers, k=2):
        super().__init__()
        if isinstance(candidates, tf.data.Dataset):
            candidates = Streaming(k=k).index(candidates, identifiers)
        self._candidates = candidates

    def call(self, queries):
        return self._candidates(queries)


# The index is filled once, then searched in a loop.
class Index:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_twice(candidates):
    index = Index()
    index.index(candidates)
    for _ in range(2):
        print(index.search(tf.ones([1, 2])))


# The class created is a parameter, so the call creating the index may create either.
class Dense:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates
        return self

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


class Scaled:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = 2 * candidates
        return self

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_with(index_class, candidates):
    index = index_class().index(candidates)
    print(index.search(tf.ones([1, 2])))
    print(index.search(tf.ones([1, 2])))


# Each iteration creates its own index, so the search it runs follows that index's only write.
class Looped:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_each(batches):
    for candidates in batches:
        index = Looped()
        index.index(candidates)
        print(index.search(tf.ones([1, 2])))
        print(index.search(tf.ones([1, 2])))


# The loop rewrites the field, but nothing reads it until the loop is over.
class Refilled:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_last(batches):
    index = Refilled()
    for candidates in batches:
        index.index(candidates)
    print(index.search(tf.ones([1, 2])))
    print(index.search(tf.ones([1, 2])))


# The index is passed to the function searching it, which is traced for each index it is given.
class Passed:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_in(index, queries):
    return tf.matmul(queries, index.candidates, transpose_b=True)


def search_passed(candidates):
    index = Passed()
    index.index(candidates)
    print(search_in(index, tf.ones([1, 2])))
    print(search_in(index, tf.ones([1, 2])))


# The function creating the holder stores the index into it, which the holder's search reads.
class Held:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


class Holding:
    def search(self, queries):
        return self.index.search(queries)


def search_held(candidates):
    holding = Holding()
    index = Held()
    index.index(candidates)
    holding.index = index
    print(holding.search(tf.ones([1, 2])))
    print(holding.search(tf.ones([1, 2])))


# Two functions each create an index of the same class and fill it before searching it.
class Twinned:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_one(candidates):
    index = Twinned()
    index.index(candidates)
    print(index.search(tf.ones([1, 2])))
    print(index.search(tf.ones([1, 2])))


def search_other(candidates):
    index = Twinned()
    index.index(candidates)
    print(index.search(tf.ones([1, 2])))
    print(index.search(tf.ones([1, 2])))


train(False)
train(True)

for i in range(2):
    retrieval = Retrieval(
        tf.data.Dataset.from_tensor_slices(tf.fill([4, 3, 2], i + 1.0)),
        tf.fill([1, 2], i + 1.0),
    )
    print(retrieval(tf.ones([1, 2])))
    print(retrieval(tf.ones([1, 2])))

search_twice(tf.fill([3, 2], 1.0))
search_twice(tf.fill([3, 2], 2.0))

search_with(Dense, tf.fill([3, 2], 1.0))
search_with(Scaled, tf.fill([3, 2], 2.0))

search_each([tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0)])
search_each([tf.fill([3, 2], 3.0)])

search_last([tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0)])
search_last([tf.fill([3, 2], 3.0)])

search_passed(tf.fill([3, 2], 1.0))
search_passed(tf.fill([3, 2], 2.0))

search_held(tf.fill([3, 2], 1.0))
search_held(tf.fill([3, 2], 2.0))

search_one(tf.fill([3, 2], 1.0))
search_other(tf.fill([3, 2], 2.0))
