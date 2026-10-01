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
    batch = pickle.load(f)

pair_sum((tf.zeros([4, 3]), tf.zeros([4, 3])))
pair_sum(batch["pair"])

pair_typed((tf.zeros([4, 3]), tf.zeros([4, 3])))
pair_typed((tf.ones([4, 3]), tf.ones([4, 3])))

pair_outer((tf.zeros([4, 3]), tf.zeros([4, 3])))
pair_outer(batch["pair"])

list_sum([tf.zeros([2]), tf.zeros([2])])
list_sum(batch["items"])

pair_splat((tf.zeros([4, 3]), tf.zeros([4, 3])))
pair_splat(**{"pair": batch["pair"]})
