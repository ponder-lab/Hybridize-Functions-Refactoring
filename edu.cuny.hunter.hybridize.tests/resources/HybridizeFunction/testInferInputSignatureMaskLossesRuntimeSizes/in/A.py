# Reduced from NLPGNN, nlpgnn/metrics/Losess.py, nlpgnn/metrics/Metric.py, nlpgnn/datas/graphloader.py, nlpgnn/gnn/messagepassing.py,
# and tests/GNN/nodes_graph_classfication/train_gan.py. The predictions come from an unsorted segment reduction whose
# `num_segments` is known only at run time, and the mask from an `np.zeros` of an ndarray's `.shape[0]`.
import pickle

import numpy as np
import tensorflow as tf


class MaskCategoricalCrossentropy:
    def __init__(self, from_logits=False, use_mask=True):
        self.from_logits = from_logits
        self.use_mask = use_mask

    def __call__(self, y_true, y_predict, input_mask=None):
        cross_entropy = tf.keras.losses.categorical_crossentropy(
            y_true, y_predict, self.from_logits
        )

        if self.use_mask:
            input_mask = tf.cast(input_mask, dtype=tf.float32)
            input_mask /= tf.reduce_mean(input_mask)
            cross_entropy *= input_mask
            return tf.reduce_mean(cross_entropy)
        else:
            return tf.reduce_mean(cross_entropy)


class MaskAccuracy:
    def __call__(self, labels, predicts, input_mask=None):
        correct_prediction = tf.equal(tf.argmax(predicts, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float64)
        mask = tf.cast(input_mask, dtype=tf.float64)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)


def masked_mean(values, input_mask):
    mask = tf.cast(input_mask, dtype=tf.float32)
    return tf.reduce_mean(values * mask)


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def aggregate(per_type, targets, num_nodes, aggr):
    if aggr == "sum":
        aggregation_fn = tf.math.unsorted_segment_sum
    elif aggr == "max":
        aggregation_fn = tf.math.unsorted_segment_max
    else:
        aggregation_fn = tf.math.unsorted_segment_mean
    messages = tf.concat(per_type, axis=0)
    message_targets = tf.concat(targets, axis=0)
    return aggregation_fn(
        data=messages, segment_ids=message_targets, num_segments=num_nodes
    )


labels = np.vstack((pickle.loads(pickle.dumps(np.eye(4))), np.eye(4)))
y_train = np.zeros(labels.shape)
y_train[:4, :] = labels[:4, :]
train_mask = sample_mask(range(4), labels.shape[0])

x = tf.linalg.matmul(tf.ones([8, 3]), tf.ones([3, 4]))
edges = tf.constant([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]])
per_type = []
for adjacency in [edges]:
    per_type.append(tf.gather(x, adjacency[:, 0]))
predict = tf.math.softmax(aggregate(per_type, [edges[:, 1]], tf.shape(x)[0], "sum"), -1)
assert predict.shape == (8, 4) and predict.dtype == tf.float32
assert train_mask.shape == (8,)

crossentropy = MaskCategoricalCrossentropy()
accscore = MaskAccuracy()
loss = crossentropy(y_train, predict, train_mask)
acc = accscore(y_train, predict, train_mask)

nodes = [n for n in range(int("8"))]
length_mask = np.zeros(len(nodes))
assert length_mask.shape == (8,)
masked_mean(tf.ones([8]), length_mask)
