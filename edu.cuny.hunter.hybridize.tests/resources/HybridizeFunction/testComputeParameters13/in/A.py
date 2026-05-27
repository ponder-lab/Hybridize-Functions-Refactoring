# Exercises `markParam`'s `default` branch (the WARN log for an unrecognized `@tf.function` kwarg). `unknown_argument` is not a real
# TF kwarg, so TF rejects it at decoration time; this fixture is not Python-runnable. The analyzer must still parse it without
# error and leave every recognized `*Param` flag unset.
import tensorflow as tf


@tf.function(unknown_argument=True)
def func(x):
    return x
