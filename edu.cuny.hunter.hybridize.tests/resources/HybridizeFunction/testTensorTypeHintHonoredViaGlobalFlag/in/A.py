# Regression fixture for #498. The function has a `tf.Tensor` type hint on its parameter
# but `experimental_follow_type_hints` is NOT supplied on the `@tf.function` decorator.
# Under the test harness's global `ALWAYS_FOLLOW_TYPE_HINTS=true`, the `followTypeHints`
# predicate inside `Parameter.classifyAsTensor` is the OR of (global flag, per-decorator
# flag), so the type hint still classifies. Pins that either flag alone suffices.

import tensorflow as tf


@tf.function
def func(x: tf.Tensor):
    pass


func(5)
