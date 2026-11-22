# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1165

# Limits: each field is written only before its first read, but in a way the check doesn't follow, so its reader stays refused.

import tensorflow as tf


# The index filled is one of two just created, selected by a condition.
class Either:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_either(flag, candidates):
    if flag:
        index = Either()
    else:
        index = Either()
    index.index(candidates)
    print(index.search(tf.ones([1, 2])))
    print(index.search(tf.ones([1, 2])))


# The method called on the index just created writes the field through a helper it calls on itself.
class Delegating:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self._store(candidates)

    def _store(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_delegating(candidates):
    index = Delegating()
    index.index(candidates)
    print(index.search(tf.ones([1, 2])))
    print(index.search(tf.ones([1, 2])))


# A function, not a method of the index, writes the field of the index just created.
class Plain:
    def __init__(self):
        self.candidates = None

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def fill(index, candidates):
    index.candidates = candidates


def search_plain(candidates):
    index = Plain()
    fill(index, candidates)
    print(index.search(tf.ones([1, 2])))
    print(index.search(tf.ones([1, 2])))


# The method is called through its class, with the index just created as its first argument.
class Unbound:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_unbound(candidates):
    index = Unbound()
    Unbound.index(index, candidates)
    print(index.search(tf.ones([1, 2])))
    print(index.search(tf.ones([1, 2])))


# The index is filled as it is created and bound once to a module's variable, through which a function searches it.
class Global:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def create_global(candidates):
    index = Global()
    index.index(candidates)
    return index


def search_global(queries):
    return global_index.search(queries)


# The index is filled as it is created, and a function nested in its creator searches it through the creator's variable.
class Enclosed:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def make_search(candidates):
    index = Enclosed()
    index.index(candidates)

    def search_enclosed(queries):
        return index.search(queries)

    return search_enclosed


search_either(True, tf.fill([3, 2], 1.0))
search_either(False, tf.fill([3, 2], 2.0))

search_delegating(tf.fill([3, 2], 1.0))
search_delegating(tf.fill([3, 2], 2.0))

search_plain(tf.fill([3, 2], 1.0))
search_plain(tf.fill([3, 2], 2.0))

search_unbound(tf.fill([3, 2], 1.0))
search_unbound(tf.fill([3, 2], 2.0))

global_index = create_global(tf.fill([3, 2], 1.0))
print(search_global(tf.ones([1, 2])))
print(search_global(tf.ones([1, 2])))

for i in range(2):
    search = make_search(tf.fill([3, 2], i + 1.0))
    print(search(tf.ones([1, 2])))
    print(search(tf.ones([1, 2])))
