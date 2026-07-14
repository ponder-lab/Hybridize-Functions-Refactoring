import numpy as np


def enumerate_param(nodes):
    # build_graph construct: enumerate the numpy-array parameter directly. (Confirmed reproduces.)
    return {int(j): i for i, j in enumerate(nodes)}


def map_dict_get(labels):
    # encode_labels construct: map a dict's `.get` over the parameter, then np.array-wrap.
    d = {0: np.zeros(3), 1: np.ones(3)}
    encoded = list(map(d.get, labels))
    return np.array(encoded, dtype=np.int32)


def map_builtin(labels):
    # Variant: map a builtin over the parameter (no dict, no np.array wrap).
    return list(map(int, labels))


def numpy_local_control(x):
    # Control: operate on a local numpy value, not the parameter (origin NUMPY -> declines).
    m = np.zeros(x.shape)
    return m + m


a = np.array([1, 2, 3])
enumerate_param(a)
map_dict_get(a)
map_builtin(a)
numpy_local_control(a)
