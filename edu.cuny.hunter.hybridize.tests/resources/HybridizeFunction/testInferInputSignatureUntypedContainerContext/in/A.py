import pickle

import tensorflow as tf


def pair_sum(pair):
    a, b = pair
    return a + b


def pair_typed(pair):
    a, b = pair
    return a + b


def pair_outer(pair):
    return pair_inner(pair)


def pair_inner(pair):
    a, b = pair
    return a + b


def list_sum(items):
    return items[0] + items[1]


def pair_splat(pair):
    a, b = pair
    return a + b


with open("batch.pkl", "rb") as f:
    p0, p1, i0, i1 = pickle.load(f)

pair = (p0, p1)
items = [i0, i1]

pair_sum((tf.zeros([4, 3]), tf.zeros([4, 3])))
pair_sum(pair)

pair_typed((tf.zeros([4, 3]), tf.zeros([4, 3])))
pair_typed((tf.ones([4, 3]), tf.ones([4, 3])))

pair_outer((tf.zeros([4, 3]), tf.zeros([4, 3])))
pair_outer(pair)

list_sum([tf.zeros([2]), tf.zeros([2])])
list_sum(items)

pair_splat((tf.zeros([4, 3]), tf.zeros([4, 3])))
pair_splat(**{"pair": pair})
