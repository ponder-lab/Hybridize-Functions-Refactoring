# Trace-time Python I/O and randomness (https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1077). Each refused
# function's Python I/O, clock read, or Python-level random draw runs once, when tf.function traces it, rather than on every call.
# The controls draw randomness, print, or compute through traced TensorFlow ops, which run on every call.
import os
import random
import tempfile
import time

import cv2
import numpy as np
import numpy as onp
import tensorflow as tf

DATA_DIR = tempfile.mkdtemp()
SRC_FILE = os.path.join(DATA_DIR, "src.txt")
ANN_FILE = os.path.join(DATA_DIR, "ann.txt")
LOG_DIR = os.path.join(DATA_DIR, "log")
PRED_FILE = os.path.join(DATA_DIR, "pred.txt")
IMG_FILE = os.path.join(DATA_DIR, "img.png")
LBL_FILE = os.path.join(DATA_DIR, "lbl.txt")

os.makedirs(LOG_DIR)

for path in (SRC_FILE, ANN_FILE):
    with open(path, "w") as out:
        out.write("x\n" * 3)

cv2.imwrite(IMG_FILE, np.full((8, 8, 3), 128, np.uint8))

with open(LBL_FILE, "w") as out:
    out.write("0 1\n1 0\n")


# OpenNMT-tf misc.count_lines, reached by both get_dataset_size overrides (opennmt/inputters/inputter.py:426, :960).
def count_lines(filename):
    with tf.io.gfile.GFile(filename, mode="rb") as f:
        num_lines = 0
        while True:
            data = f.read(65536)
            if not data:
                return num_lines
            num_lines += data.count(b"\n")


class ExampleInputter:
    def base_size(self, data_file):
        return count_lines(data_file)

    def get_dataset_size(self, x):
        size = self.base_size(SRC_FILE)
        if size != self.base_size(ANN_FILE):
            raise RuntimeError("Annotation dataset does not have the same size")
        return x * size


class ParallelInputter:
    def __init__(self, inputters):
        self.inputters = inputters

    def get_dataset_size(self, x):
        common_size = None
        for inputter, data in zip(self.inputters, (SRC_FILE, ANN_FILE)):
            size = inputter.base_size(data)
            if common_size is None:
                common_size = size
            elif size != common_size:
                raise RuntimeError("Parallel datasets do not have the same size")
        return x * tf.cast(common_size, x.dtype)


# OpenNMT-tf Model.visualize (opennmt/models/model.py:505): the inputters write projector files under the log directory.
class Inputter:
    def visualize(self, model, log_dir):
        with open(os.path.join(log_dir, "projector_config.pbtxt"), "a") as f:
            f.write("embeddings {}\n")


class Model:
    def __init__(self):
        self.features_inputter = Inputter()
        self.labels_inputter = Inputter()

    def visualize(self, x):
        self.features_inputter.visualize(self, LOG_DIR)
        self.labels_inputter.visualize(self, LOG_DIR)
        return tf.reduce_sum(x)


# OpenNMT-tf inference.predict_dataset (opennmt/inference.py:11): opens and writes a predictions file and reads the clock, and calls
# `numpy()` inside a lambda passed to `tf.nest.map_structure`, which raises under tracing.
def predict_dataset(x):
    stream = open(PRED_FILE, encoding="utf-8", mode="w")
    start_time = time.time()
    predictions = tf.nest.map_structure(lambda t: t.numpy(), {"tokens": x * 2})
    stream.write("%s\n" % predictions["tokens"])
    _ = time.time() - start_time
    stream.close()
    return x


# FCN utils.process_image_label (5-Image_Segmentation/FCN/utils.py:81): OpenCV reads the image, `random` draws the augmentation, and
# `open` reads the label.
def process_image_label(x):
    image = cv2.imread(IMG_FILE)
    gamma = random.uniform(0.8, 1.2)
    image = np.clip(image.copy() ** gamma, 0, 255)
    label = open(LBL_FILE).readlines()
    label = np.array([np.array(line.rstrip().split(" ")) for line in label], dtype=int)
    return x * image.mean() + label.sum()


# NLPGNN TUDataset.load (nlpgnn/datas/graphloader.py:511): `np.random.choice` draws the train split.
class TUDataset:
    def __init__(self):
        self.y = np.arange(20)

    def load(self, x):
        train_index = np.random.choice(np.arange(len(self.y)), size=10, replace=False)
        return tf.gather(x, train_index)


# One family each.
def reads_file(x):
    with open(SRC_FILE) as f:
        lines = f.readlines()
    return x * len(lines)


def reads_clock(x):
    return x + time.time() % 60


def draws_python_random(x):
    return x * random.random()


def draws_numpy_random(x):
    return x * np.random.uniform()


def draws_aliased_numpy_random(x):
    return x * onp.random.uniform()


def reads_image(x):
    return x * cv2.imread(IMG_FILE).mean()


def calls_reader(x):
    return reads_file(x)


def opens_through_alias(x):
    opener = open
    with opener(SRC_FILE) as f:
        lines = f.readlines()
    return x * len(lines)


# Controls.
def draws_tf_random(x):
    return x * tf.random.uniform([])


def prints_tensor(x):
    tf.print(x)
    return x * 2


def computes(x):
    return tf.reduce_sum(x * x)


class Clock:
    def time(self):
        return 1.0


def reads_user_clock(x):
    clock = Clock()
    return x * clock.time()


def calls_local_open(x):
    open = lambda path: len(path)
    return x * open(SRC_FILE)


t = tf.constant([1.0, 2.0, 3.0])
example = ExampleInputter()
assert example.get_dataset_size(t).numpy().tolist() == [3.0, 6.0, 9.0]
ParallelInputter([example, example]).get_dataset_size(t)
Model().visualize(t)
predict_dataset(t)
process_image_label(t)
TUDataset().load(tf.range(20))
reads_file(t)
reads_clock(t)
draws_python_random(t)
draws_numpy_random(t)
draws_aliased_numpy_random(t)
reads_image(t)
calls_reader(t)
opens_through_alias(t)
draws_tf_random(t)
prints_tensor(t)
computes(t)
reads_user_clock(t)
calls_local_open(t)
