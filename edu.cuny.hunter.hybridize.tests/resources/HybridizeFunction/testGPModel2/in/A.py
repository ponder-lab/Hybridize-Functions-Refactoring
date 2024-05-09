# From https://github.com/ponder-lab/GPflow/blob/8995ed0266cc55e610c967452fee5deb202a56c3/tests/gpflow/test_monitor.py#L315-L324

import tensorflow as tf


def test_compile_monitor() -> None:
    opt = tf.optimizers.Adam()

    @tf.function
    def tf_func(step: tf.Tensor) -> None:
        pass

    for step in tf.range(100):
        tf_func(step)


test_compile_monitor()
