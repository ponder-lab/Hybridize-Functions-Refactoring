import tensorflow as tf

# Reduced from the batch-tuple case of #888. A batch arriving from a generator reports on the
# parameter as the union of its elements' types, which is indistinguishable from a tensor
# parameter that several call sites type differently. Classification returned on that typing
# and never put the container question, so the elements' structure went unread and the
# parameter reduced to one flat specification where the caller passes a pair.

generator = tf.keras.preprocessing.image.ImageDataGenerator()
batches = generator.flow_from_directory("images", target_size=(8, 8))


def batched(pair):
    x, y = pair
    return tf.reduce_sum(x) + tf.reduce_sum(y)


batched(next(batches))
