# Regression fixture for #498. The function has an `int` type hint on its parameter. Even
# under the test harness's global `ALWAYS_FOLLOW_TYPE_HINTS=true`, Phase 1's
# `hasTensorTypeHint` check is specific to TF tensor types—an `int` hint does not match,
# so Phase 1 doesn't classify. The call site supplies a non-tensor, so Phase 2 doesn't fire
# either. Pins that Phase 1 is specifically TF-tensor-typed, not arbitrary-type-hinted.

import tensorflow as tf


@tf.function
def func(x: int):
    pass


func(5)
