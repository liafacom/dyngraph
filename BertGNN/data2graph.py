import logging

import numpy as np
import torch
from sklearn.preprocessing import Normalizer
from torch_geometric.data import Data


def _mean_edge_similarity(edges, vertex_features):
    if not edges:
        return 0.0
    similarities = []
    for source, target in edges:
        source_vector = vertex_features[source]
        target_vector = vertex_features[target]
        denominator = np.linalg.norm(source_vector) * np.linalg.norm(target_vector)
        similarities.append(np.dot(source_vector, target_vector) / denominator if denominator else 0.0)
    return float(np.mean(similarities))


class FAISSGraphBuilder:
    """Build a k-nearest-neighbor graph over normalized text embeddings."""

    def __init__(self, k=5, use_gpu=True, similar_min=0.3, flag_only_class=True):
        self.use_gpu = use_gpu
        self.k = k
        self.similar_min = similar_min
        self.flag_only_class = flag_only_class
        self.batch = 0

    def _build_index(self):
        import faiss

        self.index = faiss.IndexFlatL2(self.data.shape[1])
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        self.index.add(self.data)
        self.batch = 0

    def _find_neighbors(self, values, k):
        return self.index.search(values.astype("float32"), k)

    @staticmethod
    def _distance_to_similarity(distances):
        return 1 - ((distances * distances) / 2)

    def build_graph_data(self, features, labels):
        self.y = labels
        self.normalizer = Normalizer(norm="l2")
        self.data = self.normalizer.fit_transform(features)
        self._build_index()
        distances, indices = self._find_neighbors(self.data, 2 * self.k)
        similarities = self._distance_to_similarity(distances)
        self.edge_index_ = []
        self.edge_attr_ = []
        self.num_nodes = len(self.data)
        for source in range(self.num_nodes):
            self.edge_index_.append([source, source])
            self.edge_attr_.append(1)
            added = 0
            for position, target in enumerate(indices[source]):
                if added > self.k:
                    break
                same_class = self.y[source] == self.y[target]
                allowed = same_class if self.flag_only_class else True
                similarity = similarities[source][position]
                if allowed and target != source and similarity > self.similar_min:
                    self.edge_index_.extend(([source, target], [target, source]))
                    self.edge_attr_.extend((similarity, similarity))
                    added += 1
        logging.info("assortativity_coefficient base: %s", _mean_edge_similarity(self.edge_index_, self.data))

    def add_points(self, features):
        values = features if isinstance(features, np.ndarray) else features.detach().cpu().numpy()
        search = self.normalizer.transform(values)
        distances, indices = self._find_neighbors(search, self.k)
        similarities = self._distance_to_similarity(distances)
        edge_index = self.edge_index_.copy()
        edge_attr = self.edge_attr_.copy()
        for offset in range(len(values)):
            node = self.num_nodes + offset
            for position, target in enumerate(indices[offset]):
                similarity = similarities[offset][position]
                if similarity > self.similar_min:
                    edge_index.extend(([node, target], [target, node]))
                    edge_attr.extend((similarity, similarity))
        graph = Data(
            x=torch.vstack((torch.tensor(self.data, dtype=torch.float), torch.tensor(values, dtype=torch.float))),
            edge_index=torch.tensor(edge_index).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            weights=torch.tensor(edge_attr, dtype=torch.float).view(-1, 1),
            y=torch.cat([torch.tensor(self.y), -torch.ones(len(values))]),
        )
        graph.train_idx = torch.arange(self.num_nodes)
        graph.train_mask = torch.zeros(len(graph.y), dtype=torch.bool)
        graph.train_mask[graph.train_idx] = True
        graph.test_idx = torch.arange(self.num_nodes, self.num_nodes + len(values))
        graph.test_mask = torch.zeros(len(graph.y), dtype=torch.bool)
        graph.test_mask[graph.test_idx] = True
        self.batch += 1
        return graph
