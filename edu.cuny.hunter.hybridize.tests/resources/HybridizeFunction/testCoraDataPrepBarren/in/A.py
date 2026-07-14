import os

import numpy as np
import scipy.sparse as sp


class Cora(object):

    def __init__(self, extract_path="."):
        self._cora_cites = os.path.join(extract_path, "cora", "cora.cites")
        self._cora_classes = [
            "Case_Based",
            "Genetic_Algorithms",
            "Neural_Networks",
            "Probabilistic_Methods",
            "Reinforcement_Learning",
            "Rule_Learning",
            "Theory",
        ]

    def build_graph(self, nodes):
        idx_map = {int(j): i for i, j in enumerate(nodes)}
        edges_unordered = np.genfromtxt(self._cora_cites, dtype=np.int32)
        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
        ).reshape(edges_unordered.shape)
        graph = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(nodes.shape[0], nodes.shape[0]),
            dtype=np.float32,
        )
        graph += graph.T - sp.diags(graph.diagonal())
        return graph

    def sample_train_nodes(self, labels, num_per_class=20):
        train_nodes = []
        for cls in self._cora_classes:
            cls_index = np.where(labels == cls)[0]
            train_nodes += cls_index.tolist()
        return train_nodes

    def encode_labels(self, labels):
        labels_map = {}
        num_classes = len(self._cora_classes)
        for i, cls in enumerate(self._cora_classes):
            cls_label = np.zeros(shape=(num_classes,))
            cls_label[i] = 1.0
            labels_map[cls] = cls_label
        encoded_labels = list(map(labels_map.get, labels))
        return np.array(encoded_labels, dtype=np.int32)

    def split_labels(self, labels):
        num_nodes = labels.shape[0]
        encoded_labels = self.encode_labels(labels)

        def _sample_mask(index_ls):
            mask = np.zeros(num_nodes)
            mask[index_ls] = 1
            return np.array(mask, dtype=np.bool)

        def _get_labels(index_ls):
            _labels = np.zeros(encoded_labels.shape, dtype=np.int32)
            _labels[index_ls] = encoded_labels[index_ls]
            _mask = _sample_mask(index_ls)
            return _labels, _mask

        return _get_labels(self.sample_train_nodes(labels))


c = Cora()
c.build_graph(np.array([1, 2, 3]))
c.encode_labels(np.array([0, 1, 2]))
c.split_labels(np.array([0, 1, 2]))
