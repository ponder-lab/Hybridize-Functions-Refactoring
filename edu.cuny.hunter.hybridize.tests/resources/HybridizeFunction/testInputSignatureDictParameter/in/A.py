import tensorflow as tf


# The dict-valued tensor parameter: the multi-input Keras shape, and the form
# deep_recommenders' estimator rankers take as `call(self, features)`. This population had
# no fixture at all when #890 widened the container gate from sequences to every container,
# so a green suite said nothing about it. Pinned here so a change to the reduction has to
# move something visible.
def score(features):
    return tf.linalg.matmul(features["user"], features["item"], transpose_b=True)


assert score({"user": tf.ones((2, 4)), "item": tf.ones((3, 4))}).shape == (2, 3)
