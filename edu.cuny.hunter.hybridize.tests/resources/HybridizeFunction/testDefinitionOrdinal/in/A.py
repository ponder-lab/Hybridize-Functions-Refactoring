import tensorflow as tf


# Two definitions of one module-level name: ordinals 1 and 2, in source order (clause 1).
@tf.function
def twin(x):
    return x + 1


@tf.function
def twin(x):
    return x + 2


class C:
    # A method sharing the module-level name. Clause 2 counts within the qualified name, so
    # `C.twin` is its own sequence starting at 1 rather than continuing `twin` at 3.
    @tf.function
    def twin(self, x):
        return x + 3


# Defined in both arms of a branch. Clause 4 counts both, whether or not the arm executes.
if tf.executing_eagerly():

    @tf.function
    def forked(x):
        return x + 4

else:

    @tf.function
    def forked(x):
        return x + 5


@tf.function
def outer(x):
    # Nested. Clause 3 counts it; clause 2 gives it its own qualified name, so it does not
    # renumber anything at module level.
    def inner(y):
        return y + 6

    return inner(x)


@tf.function
def solo(x):
    return x + 7


# An assignment rebinding a def'd name is not a definition (clause 5), so `solo` has one
# definition rather than two.
solo = tf.function(solo)

twin(tf.constant(1))
C().twin(tf.constant(1))
outer(tf.constant(1))
solo(tf.constant(1))
