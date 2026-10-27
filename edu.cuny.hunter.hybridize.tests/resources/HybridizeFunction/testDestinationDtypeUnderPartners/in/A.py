import numpy as np
import tensorflow as tf

# The destination is allocated int64 and filled from a float64 source. NumPy casts to the
# destination on assignment, so the array stays int64 and its values are truncated. The partners
# it is passed alongside are float32. That makes the three candidate answers distinct: the
# destination allocation says int64, the partners say float32, and the assigned value says
# float64, so whichever the emission carries names its own source.
dest = np.zeros(shape=[4, 3], dtype=np.int64)
src = np.zeros(shape=[3], dtype=np.float64)
dest[0] = src

sib0 = np.zeros(shape=[4, 3], dtype=np.float32)
sib1 = np.zeros(shape=[4, 3], dtype=np.float32)


def three_way(a, b, m):
    return tf.reduce_sum(a) + tf.reduce_sum(b) + tf.reduce_sum(tf.cast(m, tf.float32))


three_way(sib0, sib1, dest)


# The same array with no partner to impose from. Partner-imposition predicts the declared int64
# here, and only the assigned value's dtype predicts float64, so this separates that hypothesis
# from the one the call above cannot rule out on its own.
alone_dest = np.zeros(shape=[4, 3], dtype=np.int64)
alone_src = np.zeros(shape=[3], dtype=np.float64)
alone_dest[0] = alone_src


def alone(m):
    return tf.reduce_sum(tf.cast(m, tf.float32))


alone(alone_dest)


# The subject's value crosses a container before landing in the destination: it is `target[2]`,
# an element of a tuple a call returned, rather than a plain array. The plain assignment above is
# read correctly, so the crossing is the next candidate for where the allocation is lost.
def make_triple():
    a = np.zeros(shape=[3], dtype=np.float64)
    b = np.zeros(shape=[3], dtype=np.float64)
    c = np.zeros(shape=[3], dtype=np.float64)
    return (a, b, c)


tup_dest = np.zeros(shape=[4, 3], dtype=np.int64)
triple = make_triple()
tup_dest[0] = triple[2]


def via_tuple(m):
    return tf.reduce_sum(tf.cast(m, tf.float32))


via_tuple(tup_dest)


# The subject crosses a second container: the destination is filled inside a generator, yielded,
# and unpacked by `next`. Kept separate from the tuple crossing so a reproduction names which one.
def gen():
    g_dest = np.zeros(shape=[4, 3], dtype=np.int64)
    g_triple = make_triple()
    g_dest[0] = g_triple[2]
    yield g_dest


g_out = next(gen())


def via_generator(m):
    return tf.reduce_sum(tf.cast(m, tf.float32))


via_generator(g_out)


# Closest mirror of the subject: the destination is allocated inside a class whose `__next__`
# returns the tuple, filled in a loop indexed by a variable rather than a constant, and unpacked
# from `next` on an instance rather than from a generator. The plain, tuple, and generator forms
# above are all read correctly, so what remains is this shape.
class Loader:
    def __iter__(self):
        return self

    def __next__(self):
        scores = np.zeros(shape=[2, 4, 3], dtype=np.float32)
        boxes = np.zeros(shape=[2, 4, 3], dtype=np.float32)
        masks = np.zeros(shape=[2, 4, 3], dtype=np.int64)

        for i in range(2):
            trip = make_triple()
            masks[i] = trip[2]

        return scores, boxes, masks


loader = Loader()
l_scores, l_boxes, l_masks = next(loader)


def via_loader(a, b, m):
    return tf.reduce_sum(a) + tf.reduce_sum(b) + tf.reduce_sum(tf.cast(m, tf.float32))


via_loader(l_scores, l_boxes, l_masks)


# How the allocation spells its dtype, which is the one thing the cases above never varied: they
# all say `np.int64`. `np.int` is the alias deprecated in NumPy 1.20 and removed in 1.24, so this
# fixture pins a NumPy below that: above it, the name does not exist and the file raises rather
# than being analyzed. Three spellings of one request, so a wrong emission names which lookup
# failed rather than only that one did.
alias_dest = np.zeros(shape=[4, 3], dtype=np.int)
i32_dest = np.zeros(shape=[4, 3], dtype=np.int32)
builtin_dest = np.zeros(shape=[4, 3], dtype=int)


def via_alias(m):
    return tf.reduce_sum(tf.cast(m, tf.float32))


def via_int32(m):
    return tf.reduce_sum(tf.cast(m, tf.float32))


def via_builtin(m):
    return tf.reduce_sum(tf.cast(m, tf.float32))


via_alias(alias_dest)
via_int32(i32_dest)
via_builtin(builtin_dest)


# Whether the alias failure is specific to `np.int` or reaches the other removed aliases. A bool
# falling back to float64 is a worse failure than an int doing so, and the corpus contains this
# spelling, so it is the one worth knowing about.
bool_alias_dest = np.zeros(shape=[4, 3], dtype=np.bool)
bool_dest = np.zeros(shape=[4, 3], dtype=np.bool_)


def via_bool_alias(m):
    return tf.reduce_sum(tf.cast(m, tf.float32))


def via_bool(m):
    return tf.reduce_sum(tf.cast(m, tf.float32))


via_bool_alias(bool_alias_dest)
via_bool(bool_dest)


# `np.bool` resolves where `np.int` does not, though both are the builtin at run time, so the gap
# is per-attribute rather than general to the removed aliases. These complete the set the corpus
# actually spells: `np.long` is int64, and `np.float` is the case where a fallback to the default
# would coincide with the intended dtype and hide the failure.
long_dest = np.zeros(shape=[4, 3], dtype=np.long)
float_alias_dest = np.zeros(shape=[4, 3], dtype=np.float)
builtin_bool_dest = np.zeros(shape=[4, 3], dtype=bool)


def via_long(m):
    return tf.reduce_sum(tf.cast(m, tf.float32))


def via_float_alias(m):
    return tf.reduce_sum(tf.cast(m, tf.float32))


def via_builtin_bool(m):
    return tf.reduce_sum(tf.cast(m, tf.float32))


via_long(long_dest)
via_float_alias(float_alias_dest)
via_builtin_bool(builtin_bool_dest)


# Neither of these is a deprecated alias, which is the point of testing them: `np.uint8` has a
# field in the model and `np.int16` does not, and int16 differs from the fallback, so it is a
# spelling whose failure is visible without being an alias at all.
u8_dest = np.zeros(shape=[4, 3], dtype=np.uint8)
i16_dest = np.zeros(shape=[4, 3], dtype=np.int16)


def via_uint8(m):
    return tf.reduce_sum(tf.cast(m, tf.float32))


def via_int16(m):
    return tf.reduce_sum(tf.cast(m, tf.float32))


via_uint8(u8_dest)
via_int16(i16_dest)
