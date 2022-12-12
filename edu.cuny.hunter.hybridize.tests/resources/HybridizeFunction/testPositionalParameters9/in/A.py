import tensorflow as tf

# experimental_compile cannot be True or False because you get the following error ValueError: Cannot specify both 'experimental_compile' and 'jit_compile'.
# That is why I put the value as None.

@tf.function(None, (tf.TensorSpec(shape=[None], dtype=tf.float32),), False, True, True, "google.matmul_low_rank_matrix", tf.autograph.experimental.Feature.EQUALITY_OPERATORS, True, None)
def test(x): 
    return x

if __name__ == '__main__':
    number = tf.constant([1.0])
    print(test(number))
