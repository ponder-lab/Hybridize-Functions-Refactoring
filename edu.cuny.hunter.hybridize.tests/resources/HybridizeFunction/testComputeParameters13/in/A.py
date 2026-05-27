# Exercises `markParam`'s `default` branch (the WARN log for an unrecognized `@tf.function` kwarg). `experimental_attributes` is a
# real TF kwarg added in TF 2.13 that Hybridize does not yet model; the analyzer must parse it without error and leave every
# recognized `*Param` flag unset. Pinning this fixture to `tensorflow==2.13.1` (newer than the project default 2.9.3) is
# intentional—the whole point is to ship a kwarg the older recognized list misses.
import tensorflow as tf


@tf.function(experimental_attributes=None)
def func(x):
    return x


if __name__ == "__main__":
    func(tf.constant(1))
