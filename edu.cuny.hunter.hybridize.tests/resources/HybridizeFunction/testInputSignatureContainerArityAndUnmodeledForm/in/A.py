import tensorflow as tf


# Lists of different lengths, one of which nests a list, so the lengths disagree and an element is a form the reduction does not model.
def nested(xs):
    return xs[0]


# Lists of different lengths, one of which holds a Python number, so an element carries no tensor evidence.
def scalar(xs):
    return xs[0]


# Lists of different lengths whose elements are all tensors, so only the lengths disagree.
def ragged(xs):
    return xs[0]


nested([tf.ones([2])])
nested([tf.ones([2]), [tf.ones([2])]])

scalar([tf.ones([2]), tf.ones([2])])
scalar([tf.ones([2]), tf.ones([2]), 3])

ragged([tf.ones([2])])
ragged([tf.ones([2]), tf.ones([2])])
