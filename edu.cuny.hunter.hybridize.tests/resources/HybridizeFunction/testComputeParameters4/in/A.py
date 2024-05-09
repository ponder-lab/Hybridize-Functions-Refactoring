import tensorflow as tf


@tf.function(experimental_implements="google.matmul_low_rank_matrix")
def func():
    pass


if __name__ == "__main__":
    func()
