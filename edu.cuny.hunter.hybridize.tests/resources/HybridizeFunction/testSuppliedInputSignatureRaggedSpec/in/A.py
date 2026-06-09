import tensorflow as tf


@tf.function(
    input_signature=[
        tf.RaggedTensorSpec(shape=[2, None], dtype=tf.int32, ragged_rank=1)
    ]
)
def func(t):
    return t


if __name__ == "__main__":
    func(tf.ragged.constant([[1, 2], [3]]))
