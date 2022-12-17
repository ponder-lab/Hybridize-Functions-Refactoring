import tensorflow as tf

@tf.function(None, (tf.TensorSpec(shape=[None], dtype=tf.float32),), True, True, "google.matmul_low_rank_matrix")
def test():
    pass

if __name__ == '__main__':
    test()
