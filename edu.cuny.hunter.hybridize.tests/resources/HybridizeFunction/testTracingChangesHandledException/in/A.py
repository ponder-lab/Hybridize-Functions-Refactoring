import tensorflow as tf

# Pins #1014 in program code: a `try` statement around a call dispatches on the exception the call
# raises, and a bare decorator can change that exception, since it traces the function with the
# argument's own shape and dtype, and an operation whose static check fails on them raises at trace
# time instead of at run time. Each refused function's guarded call is handled differently eagerly
# than under a bare decorator, and each converted one's the same, on the pinned TensorFlow 2.9.3.


# Refused: the clause catches the eager InvalidArgumentError, but tracing a rank-1 argument raises
# ValueError, which escapes it.
def square(x):
    return tf.matmul(x, x)


# Refused: its own body's `try` statement is traced along with it.
def safe_square(x):
    try:
        return square(x)
    except tf.errors.InvalidArgumentError:
        return x


safe_square(tf.ones((2, 2)))
safe_square(tf.ones((3,)))


# Converted: the index is out of range only by its value, which the kernel checks, and a call
# outside every guard passes the same shapes.
def pick(x, i):
    return tf.gather(x, i)


pick(tf.ones((5,)), tf.constant(1))

try:
    pick(tf.ones((5,)), tf.constant(7))
except tf.errors.InvalidArgumentError:
    pass


# Refused: the reverse direction. The clause lets the eager InvalidArgumentError escape, but catches
# the ValueError tracing raises.
def reverse(x):
    return tf.matmul(x, x)


reverse(tf.ones((2, 2)))


def attempt():
    try:
        reverse(tf.ones((3,)))
    except ValueError:
        return "ValueError"
    return "returned"


try:
    attempt()
except tf.errors.InvalidArgumentError:
    pass


# Converted: one clause catches both.
def caught_either_way(x):
    return tf.matmul(x, x)


caught_either_way(tf.ones((2, 2)))

try:
    caught_either_way(tf.ones((3,)))
except (tf.errors.InvalidArgumentError, ValueError):
    pass


# Converted: a bare `except` catches every exception.
def caught_by_bare(x):
    return tf.matmul(x, x)


caught_by_bare(tf.ones((2, 2)))

try:
    caught_by_bare(tf.ones((3,)))
except:  # noqa: E722
    pass


# Converted: a call in the `else` clause is not guarded by the statement, so the clause decides
# nothing about it.
def in_else(x):
    return tf.matmul(x, x)


try:
    in_else(tf.ones((2, 2)))
except tf.errors.InvalidArgumentError:
    pass
else:
    in_else(tf.ones((2, 2)))


# Converted: the innermost clause to catch either error catches both, so the outer one never sees
# them.
def caught_inside(x):
    return tf.matmul(x, x)


caught_inside(tf.ones((2, 2)))

try:
    try:
        caught_inside(tf.ones((3,)))
    except (tf.errors.InvalidArgumentError, ValueError):
        pass
except tf.errors.InvalidArgumentError:
    pass


# Refused: a clause naming its classes through a name the module assigns catches what the name is
# assigned, here the eager error alone.
ERRORS = (tf.errors.InvalidArgumentError,)


def named_clause(x):
    return tf.matmul(x, x)


named_clause(tf.ones((2, 2)))

try:
    named_clause(tf.ones((3,)))
except ERRORS:
    pass


# Refused: a clause whose classes are computed is not read, so what it catches is unknown.
def errors_of():
    return (tf.errors.InvalidArgumentError,)


def computed_clause(x):
    return tf.matmul(x, x)


computed_clause(tf.ones((2, 2)))

try:
    computed_clause(tf.ones((3,)))
except errors_of():
    pass


# Converted: a function defined inside a `try` statement is called outside it, so its body is not
# guarded by the statement.
try:

    def defined_in_try(x):
        return tf.matmul(x, x)

except tf.errors.InvalidArgumentError:
    pass

defined_in_try(tf.ones((2, 2)))

SCALE = "2"


# Converted: its own `try` statement guards a builtin, which raises no TensorFlow error.
def scaled(x):
    try:
        factor = float(SCALE)
    except ValueError:
        factor = 1.0
    return x * factor


scaled(tf.ones((2,)))
