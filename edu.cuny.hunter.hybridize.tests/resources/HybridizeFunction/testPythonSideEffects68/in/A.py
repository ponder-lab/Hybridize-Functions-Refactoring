import tensorflow as tf


def compute(dataset):
    def scale(element):
        return element * 2.0

    def top_pair(element):
        scores = tf.matmul(element, element)
        return (scores, scores)

    dataset = dataset.map(scale)
    dataset = dataset.map(top_pair)
    return dataset


ds = tf.data.Dataset.from_tensor_slices([[[1.0, 2.0], [3.0, 4.0]]])
result = compute(ds)
for pair in result:
    pass
