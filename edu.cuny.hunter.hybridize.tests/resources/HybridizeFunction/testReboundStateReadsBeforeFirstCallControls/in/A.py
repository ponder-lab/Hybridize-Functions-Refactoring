# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1165

# Controls: each field is rewritten after something has read it, or its object is reached through a variable, a field, or a container
# that is rebound, so a traced read is stale.

from dataclasses import dataclass
from typing import Any, NamedTuple

import tensorflow as tf


# The function creating the index fills it again after a search.
class Reindexed:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_then_reindex(first, second):
    index = Reindexed()
    index.index(first)
    print(index.search(tf.ones([1, 2])))
    index.index(second)
    print(index.search(tf.ones([1, 2])))


# The index is filled by a function receiving it as an argument, which may be called at any time.
class Shared:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def reindex(index, candidates):
    index.index(candidates)


def search_shared(first, second):
    index = Shared()
    reindex(index, first)
    print(index.search(tf.ones([1, 2])))
    reindex(index, second)
    print(index.search(tf.ones([1, 2])))


# The index is a module's variable, which the function filling it doesn't create.
class Registered:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


# Each index is filled as it is created, but the function searching reaches it through a variable it declares global, which another
# function rebinds. The index's own method is traced per index, so its reads aren't stale.
class Declared:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def renew_declared(candidates):
    global declared
    index = Declared()
    index.index(candidates)
    declared = index


def search_declared(queries):
    global declared
    return declared.search(queries)


# The method called on the index just created was rebound on it to the method of the index created before it, which has been searched.
class Relooped:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_relooped(first, second):
    previous = None
    for step in range(2):
        index = Relooped()
        if previous is None:
            index.index(first)
            print(index.search(tf.ones([1, 2])))
            previous = index
        else:
            index.index = previous.index
            index.index(second)
            print(previous.search(tf.ones([1, 2])))


# The method called on the tuple just created is an index's, which the tuple's constructor stored in its field.
class Packed:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


class Carton(NamedTuple):
    pack: Any


def fill_through_carton(index, candidates):
    carton = Carton(index.index)
    carton.pack(candidates)


def search_packed(first, second):
    index = Packed()
    fill_through_carton(index, first)
    print(index.search(tf.ones([1, 2])))
    fill_through_carton(index, second)
    print(index.search(tf.ones([1, 2])))


# The method called on the index just created is the one its constructor stored in another field: that of the index created before it,
# which has been searched.
@dataclass(eq=False)
class Dated:
    fill: Any = None
    candidates: Any = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_dated(first, second):
    previous = None
    for step in range(2):
        index = Dated(None if previous is None else previous.index)
        if previous is None:
            index.index(first)
            print(index.search(tf.ones([1, 2])))
            previous = index
        else:
            index.fill(second)
            print(previous.search(tf.ones([1, 2])))


# The method called on the index just created is one its class holds in an attribute, which the class's body sets to a method of another
# object, as a call's result, and the program then to that of the index created before it, which a function given it has searched.
class Spare:
    def index(self, candidates):
        pass


def spare_index():
    return Spare().index


class Classed:
    fill = spare_index()

    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates


def search_classed(index, queries):
    return tf.matmul(queries, index.candidates, transpose_b=True)


def fill_classed(first, second):
    previous = None
    for step in range(2):
        index = Classed()
        if previous is None:
            index.index(first)
            print(search_classed(index, tf.ones([1, 2])))
            previous = index
            Classed.fill = previous.index
        else:
            index.fill(second)
            print(search_classed(previous, tf.ones([1, 2])))


# The same, where the class's body names the other object's method itself.
class Aliased:
    fill = Spare().index

    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates


def search_aliased(index, queries):
    return tf.matmul(queries, index.candidates, transpose_b=True)


def fill_aliased(first, second):
    previous = None
    for step in range(2):
        index = Aliased()
        if previous is None:
            index.index(first)
            print(search_aliased(index, tf.ones([1, 2])))
            previous = index
            Aliased.fill = previous.index
        else:
            index.fill(second)
            print(search_aliased(previous, tf.ones([1, 2])))


# The method called on the index just created was replaced on its class by that of the index created before it, which a function given
# it has searched.
class Renamed:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates


def search_renamed(index, queries):
    return tf.matmul(queries, index.candidates, transpose_b=True)


def fill_renamed(first, second):
    previous = None
    for step in range(2):
        index = Renamed()
        if previous is None:
            index.index(first)
            print(search_renamed(index, tf.ones([1, 2])))
            previous = index
            Renamed.index = previous.index
        else:
            index.index(second)
            print(search_renamed(previous, tf.ones([1, 2])))


registered = Registered()


def register(candidates):
    registered.index(candidates)


# The index is held in a field of the object whose method fills it.
class Owned:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


class Owner:
    def __init__(self):
        self.index = Owned()

    def refill(self, candidates):
        self.index.index(candidates)

    def search(self, queries):
        return self.index.search(queries)


# The constructor searches the index, so the search's trace precedes the write that follows the construction.
class Searching:
    def __init__(self, candidates):
        self.candidates = candidates
        print(self.search(tf.ones([1, 2])))

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_after_construction(first, second):
    index = Searching(first)
    index.index(second)
    print(index.search(tf.ones([1, 2])))


# The method filling the index searches it first.
class Probing:
    def __init__(self, candidates):
        self.candidates = candidates

    def index(self, candidates):
        print(self.search(tf.ones([1, 2])))
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_after_probe(first, second):
    index = Probing(first)
    index.index(second)
    print(index.search(tf.ones([1, 2])))


# The method called on the box just created is another object's, bound to an index that outlives the call.
class Borrowed:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


class Box:
    pass


def fill_through_box(index, candidates):
    box = Box()
    box.fill = index.index
    box.fill(candidates)


def search_borrowed(first, second):
    index = Borrowed()
    fill_through_box(index, first)
    print(index.search(tf.ones([1, 2])))
    fill_through_box(index, second)
    print(index.search(tf.ones([1, 2])))


# The method called on the index just created was rebound on it to another index's, which has been searched.
class Patched:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_patched(first, second):
    index = Patched()
    index.index(first)
    print(index.search(tf.ones([1, 2])))
    other = Patched()
    other.index = index.index
    other.index(second)
    print(index.search(tf.ones([1, 2])))


# The generator creating the index hands it out between two fills, so a search runs between them.
class Stepped:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def fill_in_steps(first, second):
    index = Stepped()
    index.index(first)
    yield index
    index.index(second)
    yield index


def search_stepped(first, second):
    for index in fill_in_steps(first, second):
        print(index.search(tf.ones([1, 2])))


# Each index is filled as it is created, but its keeper replaces the index it holds between two searches. The index's own method is
# traced per index, so its reads aren't stale.
class Kept:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


class Keeper:
    def __init__(self):
        self.index = None

    def replace(self, candidates):
        index = Kept()
        index.index(candidates)
        self.index = index

    def search(self, queries):
        return self.index.search(queries)


# Each index is filled as it is created, but the search reads the last one of a list that grows between two searches.
class Listed:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


class Lister:
    def __init__(self):
        self.indexes = []

    def add(self, candidates):
        index = Listed()
        index.index(candidates)
        self.indexes.append(index)

    def search(self, queries):
        return self.indexes[-1].search(queries)


# Each index is filled as it is created, but the search reads it from a tuple that is replaced between two searches.
class Paired:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


class Pairer:
    def __init__(self):
        self.pair = None

    def replace(self, candidates):
        index = Paired()
        index.index(candidates)
        self.pair = (index, candidates)

    def search(self, queries):
        return self.pair[0].search(queries)


# Each index is filled as it is created, but the function searching reaches it through a module's variable, which is rebound. The method
# is traced per index, so its own reads aren't stale.
class Current:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_current(queries):
    return current.search(queries)


# The function creating the index sets its field itself again after a search.
class Rewritten:
    def __init__(self):
        self.candidates = None

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_then_rewrite(first, second):
    index = Rewritten()
    index.candidates = first
    print(index.search(tf.ones([1, 2])))
    index.candidates = second
    print(index.search(tf.ones([1, 2])))


# A function receiving the index as an argument sets its field, and may be called at any time.
class Assigned:
    def __init__(self):
        self.candidates = None

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def assign(index, candidates):
    index.candidates = candidates


def search_assigned(first, second):
    index = Assigned()
    assign(index, first)
    print(index.search(tf.ones([1, 2])))
    assign(index, second)
    print(index.search(tf.ones([1, 2])))


# The generator creating the index sets its field itself between handing it out twice, so a search runs between the two writes.
class Yielded:
    def __init__(self):
        self.candidates = None

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def set_in_steps(first, second):
    index = Yielded()
    index.candidates = first
    yield index
    index.candidates = second
    yield index


def search_yielded(first, second):
    for index in set_in_steps(first, second):
        print(index.search(tf.ones([1, 2])))


# The function filling the index gets it from a call that hands out the same index each time.
class Fetched:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


fetched = Fetched()


def fetch():
    return fetched


def refill_fetched(candidates):
    index = fetch()
    index.index(candidates)


# Each index is filled as it is created, but the nested function searching reaches it through its enclosing function's variable,
# which that function rebinds. The index's own method is traced per index, so its reads aren't stale.
class Renewed:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


def search_renewed(first, second):
    index = Renewed()
    index.index(first)

    def search_enclosing(queries):
        return index.search(queries)

    print(search_enclosing(tf.ones([1, 2])))
    index = Renewed()
    index.index(second)
    print(search_enclosing(tf.ones([1, 2])))


# Each index is filled as it is created and handed to its carrier's constructor, but the function searching reaches the carrier
# through a module's variable, which is rebound. The index's own method is traced per index, so its reads aren't stale.
class Carried:
    def __init__(self):
        self.candidates = None

    def index(self, candidates):
        self.candidates = candidates

    def search(self, queries):
        return tf.matmul(queries, self.candidates, transpose_b=True)


class Carrier:
    def __init__(self, index):
        self.index = index


def make_carrier(candidates):
    index = Carried()
    index.index(candidates)
    return Carrier(index)


def search_carried(queries):
    return carrier.index.search(queries)


search_then_reindex(tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0))

search_shared(tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0))

for i in range(2):
    register(tf.fill([3, 2], i + 1.0))
    print(registered.search(tf.ones([1, 2])))

owner = Owner()
for i in range(2):
    owner.refill(tf.fill([3, 2], i + 1.0))
    print(owner.search(tf.ones([1, 2])))

search_after_construction(tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0))

search_after_probe(tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0))

search_borrowed(tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0))

search_patched(tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0))

search_stepped(tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0))

keeper = Keeper()
for i in range(2):
    keeper.replace(tf.fill([3, 2], i + 1.0))
    print(keeper.search(tf.ones([1, 2])))

lister = Lister()
for i in range(2):
    lister.add(tf.fill([3, 2], i + 1.0))
    print(lister.search(tf.ones([1, 2])))

pairer = Pairer()
for i in range(2):
    pairer.replace(tf.fill([3, 2], i + 1.0))
    print(pairer.search(tf.ones([1, 2])))

for i in range(2):
    current = Current()
    current.index(tf.fill([3, 2], i + 1.0))
    print(search_current(tf.ones([1, 2])))

search_then_rewrite(tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0))

search_assigned(tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0))

search_yielded(tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0))

for i in range(2):
    refill_fetched(tf.fill([3, 2], i + 1.0))
    print(fetched.search(tf.ones([1, 2])))

search_renewed(tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0))

for i in range(2):
    carrier = make_carrier(tf.fill([3, 2], i + 1.0))
    print(search_carried(tf.ones([1, 2])))

for i in range(2):
    renew_declared(tf.fill([3, 2], i + 1.0))
    print(search_declared(tf.ones([1, 2])))

search_relooped(tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0))

search_packed(tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0))

search_dated(tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0))

fill_classed(tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0))

fill_aliased(tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0))

fill_renamed(tf.fill([3, 2], 1.0), tf.fill([3, 2], 2.0))
