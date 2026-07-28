import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from sklearn import preprocessing
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from tqdm import tqdm
from torch_geometric.utils import to_networkx
from collections import Counter
import logging


def imprimir_frequencia(lista):
    frequencia = Counter(lista)
    
    for item, freq in frequencia.items():
        print(f'{item}: {freq}')
        
plots = 0

def calculate_heterophily_edges(edges, labels):
    """
    Calcula a heterofilia do grafo.
    
    Parameters:
    - graph: um objeto NetworkX representando o grafo.
    - labels: um dicionário que mapeia nós para seus respectivos rótulos.
    
    Returns:
    - A fração de arestas que conectam nós com rótulos diferentes.
    """
    heterophilic_edges = 0
    total_edges = len(edges)
    
    for edge in edges:
        node1, node2 = edge
        if labels[node1] != labels[node2]:
            heterophilic_edges += 1
    
    return heterophilic_edges / total_edges if total_edges > 0 else 0

def calculate_heterophily(graph, labels):
    """
    Calcula a heterofilia do grafo.
    
    Parameters:
    - graph: um objeto NetworkX representando o grafo.
    - labels: um dicionário que mapeia nós para seus respectivos rótulos.
    
    Returns:
    - A fração de arestas que conectam nós com rótulos diferentes.
    """
    heterophilic_edges = 0
    total_edges = graph.number_of_edges()
    
    for edge in graph.edges():
        node1, node2 = edge
        if labels[node1] != labels[node2]:
            heterophilic_edges += 1
    
    return heterophilic_edges / total_edges if total_edges > 0 else 0

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

def cal_sim_homog(edges, vertex_features):
    # Crie o grafo
    G = nx.Graph()
    for n in range(len(vertex_features)):
        G.add_node(n)
    for u, v in edges:
        G.add_edge(u, v)
    
    # Atribua as características aos vértices
    for node, features in enumerate(vertex_features):
        G.nodes[node]['features'] = features

    # Função para calcular a similaridade média do cosseno nas arestas
    # Calcular a similaridade média do cosseno
    graph = G
    similarities = []
    for u, v in graph.edges():
        vec_u = np.array(graph.nodes[u]['features']).reshape(1, -1)
        vec_v = np.array(graph.nodes[v]['features']).reshape(1, -1)
        similarity = cosine_similarity(vec_u, vec_v)[0][0]
        similarities.append(similarity)
    return np.mean(similarities)


def plot_graph(data, name_fig="graph", i=None):
    global plots
    if i is None:
        plots+=1
    G = nx.Graph()
    G = to_networkx(data, node_attrs=['x', 'y', 'train_mask'], edge_attrs=['edge_attr'])
    # Adicionando vértices e arestas ao grafo
    # for idx, tag in enumerate(data.type_node):
    #     G.add_node(idx)
    colors = [  
        'red', 'blue', 'green', 'purple', 
        'orange', 'brown', 'pink', 'gray', 'olive', 'cyan',
        'magenta', 'yellow', 'lightblue', 'lightgreen', 'lightcoral', 'darkorange', 'saddlebrown',
        'lightpink', 'darkgray', 'darkolivegreen', 'teal', 'gold', 'darkviolet', 'lime', 
        'slateblue', 'darkred', 'navy', 'springgreen', 
        'red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan',
        'magenta', 'yellow', 'lightblue', 'lightgreen', 'lightcoral', 'darkorange', 'saddlebrown',
        'lightpink', 'darkgray', 'darkolivegreen', 'teal', 'gold', 'darkviolet', 'lime', 
        'slateblue', 'darkred', 'navy', 'springgreen', 
    ]
    # Mapeamento de cores para os vértices
    vertex_colors = {i: c for i, c in enumerate(colors)}
    vertex_colors.update({-1: 'black'})
    
    edge_colors = {'pos_tag': 'magenta', 'tfidf': 'gold', 'pmi': 'cyan'}
    
    # for i, (u, v) in enumerate(data.edge_index.t()):
    #     if i == 0:
    #         print(u.item(),v.item())
    #     G.add_edge(u.item(), v.item(), color=vertex_colors[data.y[v.item()]])
    if isinstance(data.y, list):
        y_list = data.y
    else:
        y_list = data.y.tolist()
    def get_color(u, v):
        u = int(u)
        v = int(v)
        if y_list[v] == y_list[u]:
            return vertex_colors[y_list[v]] 
        elif y_list[v] == -1 or y_list[u] == -1:
            return 'lightgray'
        else:
            return 'black'
    
    node_color = [vertex_colors[y_list[i]]  for i in range(len(data.x))]
    edge_color = [get_color(u, v) for u, v in G.edges]
    imprimir_frequencia(edge_color)
    # nx.draw(G, pos, 
    #         # with_labels=True, 
    #         node_color=node_color,
    #         edge_color=edge_color, node_size=25, font_size=8)
    # Coletar os nós com 'train_mask' True e False
    return
    # Configurações de cores e formas
    train_shape = 'o'  # Triângulo
    test_shape = 'v'  # Círculo
    # import ipdb; ipdb.set_trace()
    # Desenhar os nós baseado nos atributos
    # Desenhando o grafo
    pos = nx.spring_layout(G, k=1, weight='edge_attr')
    for node, data in G.nodes(data=True):
        node_color = vertex_colors[data['y']]
        if 'train_mask' in data:
            node_shape = train_shape if data['train_mask'] else test_shape
        else:
            node_shape = train_shape
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=node_color, node_shape=node_shape, node_size=25)
    # Desenhar as arestas com cores baseadas nos atributos
    # edge_colors = nx.get_edge_attributes(G, 'color')
    edges = G.edges()
    # colors = [edge_colors[edge] if edge in edge_colors else 'gray' for edge in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_color, node_size=25)

    # edge_labels = nx.get_edge_attributes(G, "edge_attr")
    # edge_labels = {k: round(v, 2) for k, v in edge_labels.items()}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels)
    append = ""
    if i:
        append = f"_{i}"
        
    plt.savefig(f"{name_fig}_{plots}{append}.png")
    plt.close()


class FAISSGraphBuilder:
    def __init__(self, k=5, use_gpu=True, similar_min=0.3, flag_only_class=True):
        # self.data = data.astype(
        #     "float32"
        # )  # Garantindo que os dados estejam no dtype correto
        # self.data /= np.linalg.norm(self.data, axis=1, keepdims=True)
        self.use_gpu = use_gpu
        self.index = None
        self.k = k
        self.similar_min = similar_min
        self.flag_only_class = flag_only_class
        self.batch = 0
        self.X = None
        # self._build_index()

    def _build_index(self):
        import faiss

        d = self.data.shape[1]
        if self.use_gpu and faiss.get_num_gpus() > 0:
            print("clean index")
            self.index = faiss.IndexFlatL2(d)
            self.index = faiss.index_cpu_to_all_gpus(
                self.index
            )  # Movendo o índice para a GPU, se disponível
        else:
            self.index = faiss.IndexFlatL2(d)
        self.index.add(self.data)
        self.batch = 0

    def find_k_neighbors(self, X, k):
        X = X.astype(
            "float32"
        )  # Garantindo que os vetores de consulta estejam no dtype correto
        distances, indices = self.index.search(X, k)
        return distances, indices

    def build_graph_data(self, X_, y_true):
        # import ipdb;ipdb.set_trace()
        
        self.y = y_true
        self.normalizer = Normalizer(norm='l2')  # norm='l2' para L2 normalization
        self.X = X_
        self.data = self.normalizer.fit_transform(X_)
        # mean_vec_class = {}
        # for cl in np.unique(y_true):
        #     label_vectors = self.data[y_true == cl]
        #     mean_vec_class[cl] = np.mean(label_vectors, axis=0)
        # self.data = self.data.cpu().numpy()
        self._build_index()
        X = self.data
        distances, indices = self.find_k_neighbors(X, 2*self.k)
        # if distances.any() > 100000:
        #     distances[distances > 100000] = 100000
        distances = 1 - ((distances * distances) / 2)
        self.edge_index_ = []
        self.edge_attr_ = []
        self.num_nodes = X.shape[0]

        for i in range(self.num_nodes):
            self.edge_index_.append([i, i])
            self.edge_attr_.append(1)
            added = 0
            for j, neighbor_index in enumerate(indices[i]):
                if added > self.k:
                    break
                if self.flag_only_class:
                    allow_to_add = self.y[i] == self.y[neighbor_index]
                else:
                    allow_to_add = True
                if allow_to_add and (neighbor_index != i and distances[i][j] > self.similar_min):
                    self.edge_index_.append([i, neighbor_index])
                    self.edge_index_.append([neighbor_index, i])
                    self.edge_attr_.append(distances[i][j])
                    self.edge_attr_.append(distances[i][j])
                    added += 1
                    
        
        # data = Data(x=X, edge_index=torch.tensor(self.edge_index_).t().contiguous(), 
        #             edge_attr=self.edge_attr_,
        #             y=list(self.y), 
        #             train_mask=[True] * len(X),
        #             type_node=["doc"] * len(X), 
        #             type_edge=["tfidf"] * len(self.edge_index_))  
        # plot_graph(data, name_fig="graph")     
        # logging.info(f"heterophily base: {calculate_heterophily_edges(self.edge_index_, y_true)}")
        
        # Crie o grafo
        logging.info(f"assortativity_coefficient base: {cal_sim_homog(self.edge_index_, X)}")
        

    def get_data(self):

        edge_index = torch.tensor(self.edge_index_).t().contiguous()
        edge_attr = torch.tensor(self.edge_attr_, dtype=torch.float).view(-1, 1)

        data = Data(
            x=torch.tensor(self.X, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            weights=edge_attr,
            # y=y,
        )
        
        return data
    
    def add_val_test(self, X_val, X_test):
        l_train = len(self.data)
        len_val = len(X_val)
        len_test = len(X_test)
        self.val_idx = torch.arange(l_train, l_train + len_val)
        self.test_idx = torch.arange(l_train + len_val, l_train + len_val + len_test)
        
        X = np.concatenate([X_val, X_test])    
        
        search = X / np.linalg.norm(X, axis=1, keepdims=True)
        # search = search.cpu().numpy()
        distances, indices = self.find_k_neighbors(search, self.k)
        distances = 1 - ((distances * distances) / 2)
        self.new_nodes = len(X)

        self.edge_index = self.edge_index_.copy()
        self.edge_attr = self.edge_attr_.copy()

        for i in range(self.new_nodes):
            # print(i, self.num_nodes + i, indices[i])
            for j, neighbor_index in enumerate(indices[i]):
                if distances[i][j] > self.similar_min:
                    # print(self.num_nodes + i, neighbor_index)
                    self.edge_index.append([self.num_nodes + i, neighbor_index])
                    self.edge_index.append([neighbor_index, self.num_nodes + i])
                    self.edge_attr.append(distances[i][j])
                    self.edge_attr.append(distances[i][j])
        
        return self.generate_data_val_test(X)
    
    def generate_data_val_test(self, X):
        edge_index = torch.tensor(self.edge_index).t().contiguous()
        edge_attr = torch.tensor(self.edge_attr, dtype=torch.float).view(-1, 1)

        features = torch.vstack(
            (
                torch.tensor(self.data, dtype=torch.float),
                torch.tensor(X, dtype=torch.float),
            )
        )
        y = torch.cat([torch.tensor(self.y, dtype=torch.long), 
                       -1 * torch.ones(self.new_nodes, dtype=torch.long)])
        data = Data(
            x=features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            weights=edge_attr,
            y=y,
        )
        l_train = len(self.data)
        # Add additional arguments to `data`:
        data.train_idx = torch.tensor(np.arange(l_train), dtype=torch.long)

        a = torch.zeros(len(data.y), dtype=torch.float)
        a[data.train_idx] = True
        data.train_mask = a

        data.val_idx = torch.tensor(self.val_idx, dtype=torch.long)
        a = torch.zeros(len(data.y), dtype=torch.float)
        a[data.val_idx] = True
        data.val_mask = a

        data.test_idx = torch.tensor(self.test_idx, dtype=torch.long)
        a = torch.zeros(len(data.y), dtype=torch.float)
        a[data.test_idx] = True
        data.test_mask = a
        
        return data

    def add_points_batch(self, X, n_batch=8):
        if not isinstance(X, np.ndarray):
            temp = X.detach().cpu().numpy()
        else:
            temp = X
        search = temp / np.linalg.norm(temp, axis=1, keepdims=True)
        # search = search.cpu().numpy()
        distances, indices = self.find_k_neighbors(search, self.k)
        distances = 1 - ((distances * distances) / 2)
        
        batches = []

        # Calculando o número de batches necessário
        num_batches = (distances.shape[0] + n_batch - 1) // n_batch

        # Dividindo a matriz em batches
        dists = [distances[i * n_batch:(i + 1) * n_batch] for i in range(num_batches)]
        indices_ = [indices[i * n_batch:(i + 1) * n_batch] for i in range(num_batches)]
        
        for dist, ind in tqdm(zip(dists, indices_), total=len(dists)):
            self.new_nodes = len(dist)

            self.edge_index = self.edge_index_.copy()
            self.edge_attr = self.edge_attr_.copy()

            for i in range(self.new_nodes):
                # print(i, self.num_nodes + i, indices[i])
                for j, neighbor_index in enumerate(ind[i]):
                    if dist[i][j] > self.similar_min:
                        # print(self.num_nodes + i, neighbor_index)
                        self.edge_index.append([self.num_nodes + i, neighbor_index])
                        self.edge_index.append([neighbor_index, self.num_nodes + i])
                        self.edge_attr.append(dist[i][j])
                        self.edge_attr.append(dist[i][j])
            batches.append(self.generate_data(temp))
            
        return batches
    
    def add_points(self, X, y_true=None):
        
        if not isinstance(X, np.ndarray):
            temp = X.detach().cpu().numpy()
        else:
            temp = X
        search = self.normalizer.transform(temp) 
        # search = search.cpu().numpy()
        # import ipdb; ipdb.set_trace()
        distances, indices = self.find_k_neighbors(search, self.k)
        distances = 1 - ((distances * distances) / 2)
        self.new_nodes = len(X)

        self.edge_index = self.edge_index_.copy()
        self.edge_attr = self.edge_attr_.copy()

        for i in range(self.new_nodes):
            # print(i, self.num_nodes + i, indices[i])
            for j, neighbor_index in enumerate(indices[i]):
                if distances[i][j] > self.similar_min:
                    # print(self.num_nodes + i, neighbor_index)
                    self.edge_index.append([self.num_nodes + i, neighbor_index])
                    self.edge_index.append([neighbor_index, self.num_nodes + i])
                    self.edge_attr.append(distances[i][j])
                    self.edge_attr.append(distances[i][j])

        # data = Data(x=X, edge_index=torch.tensor(self.edge_index_).t().contiguous(), 
        #             edge_attr=self.edge_attr_,
        #             y=list(self.y), 
        #             type_node=["doc"] * len(X), 
        #             type_edge=["tfidf"] * len(self.edge_index_))  
        
        data = self.generate_data(temp, y_true)
           
        self.batch +=1
        if self.batch == 1:
            G = to_networkx(data, node_attrs=['x', 'y', 'train_mask'], edge_attrs=['edge_attr'])
            logging.info(f"heterophily: {calculate_heterophily(G, data.y.numpy())}")
        # plot_graph(data, name_fig="graph", i=self.batch)  
        
        return data

    def generate_data(self, X, y_train=None):
        edge_index = torch.tensor(self.edge_index).t().contiguous()
        edge_attr = torch.tensor(self.edge_attr, dtype=torch.float)
        weights = torch.tensor(self.edge_attr, dtype=torch.float).view(-1, 1)

        features = torch.vstack(
            (
                torch.tensor(self.data, dtype=torch.float),
                torch.tensor(X, dtype=torch.float),
            )
        )
        if y_train:
            y = torch.cat([torch.tensor(self.y), torch.tensor(y_train)])
        else:
            y = torch.cat([torch.tensor(self.y), -1 * torch.ones(self.new_nodes)])
        data = Data(
            x=features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            weights=weights,
            y=y,
        )

        l_train = len(self.data)
        # Add additional arguments to `data`:
        data.train_idx = torch.arange(l_train)

        a = torch.zeros(len(data.y), dtype=torch.bool)
        a[data.train_idx] = True
        data.train_mask = a

        if X is not None:
            data.test_idx = torch.arange(l_train, l_train + self.new_nodes)
            a = torch.zeros(len(data.y), dtype=torch.bool)
            a[data.test_idx] = True
            data.test_mask = a
        
        return data

    def adding_data(self, X_train, y_train, X_test):
        features = X_train
        y_true = torch.tensor(y_train)
        if X_test is not None:
            l_test = len(X_test)
            features = torch.vstack((features, X_test))
            y_true = torch.cat([y_true, -1 * torch.ones(l_test)])

        data = self.build_graph_data(features, y_true, self.k)
        l_train = len(X_train)
        # Add additional arguments to `data`:
        data.train_idx = torch.arange(l_train)

        a = torch.zeros(len(data.y), dtype=torch.bool)
        a[data.train_idx] = True
        data.train_mask = a

        if X_test is not None:
            l_test = len(X_test)
            data.test_idx = torch.arange(l_train, l_train + l_test)
            a = torch.zeros(len(data.y), dtype=torch.bool)
            a[data.test_idx] = True
            data.test_mask = a
        return data


class Data2Graph:
    def __init__(
        self,
        k_neigs=7,
        mode="connectivity",
        transductive=False,
        normalize=False,
        scaled_features=False,
        verbose=False,
        adaptative=False,
    ) -> None:
        self.k_neigs = k_neigs
        self.mode = mode
        self.transductive = transductive
        self.verbose = verbose
        self.normalize = normalize
        self.scaled_features = scaled_features
        self.adaptative = adaptative

    def operar_matriz_esparsa(self, matriz_esparsa, k):
        # Encontrando os índices dos elementos não-nulos
        nnz_row_indices, nnz_col_indices = matriz_esparsa.nonzero()

        # Realizando a operação apenas nos elementos não-nulos
        for i, j in zip(nnz_row_indices, nnz_col_indices):
            # k / matriz_esparsa[i, j]
            matriz_esparsa[i, j] = 1 - (matriz_esparsa[i, j] / 2)

        # matriz_esparsa = matriz_esparsa/matriz_esparsa.max()
        if self.normalize is False:
            return matriz_esparsa
        norma = norm(matriz_esparsa)

        if norma == 0:
            return matriz_esparsa
        else:
            return matriz_esparsa / norma

    def fit(self, X_train, y_train, X_val, y_val=None):
        if self.transductive:
            return self.fit_transform_(X_train, y_train, X_val, y_val, X_test=None)
        else:
            return self.fit_transform_(
                X_train, y_train, X_val=None, y_val=None, X_test=None
            )

    def transform(self, X_train, y_train, X_test=None, X_val=None, y_val=None):
        if self.transductive:
            return self.fit_transform_(
                X_train, y_train, X_val=X_val, y_val=y_val, X_test=X_test
            )
        else:
            return self.fit_transform_(
                X_train, y_train, X_val=None, y_val=None, X_test=X_test
            )

    def fit_transform_(self, X_train, y_train, X_val, y_val=None, X_test=None):

        l_train = len(X_train)
        features = X_train
        y_true_numeric = y_train

        l_test = l_val = 0
        if X_val is not None:
            l_val = len(X_val)
            features = np.vstack((X_train, X_val))

            if y_val is None:
                y_true_numeric = np.concatenate([y_train, -1 * np.ones(l_val)])
            else:
                y_true_numeric = np.concatenate([y_train, y_val])

        if X_test is not None:
            l_test = len(X_test)
            features = np.vstack((features, X_test))
            y_true_numeric = np.concatenate([y_true_numeric, -1 * np.ones(l_test)])

        if self.scaled_features:
            self.scaler = preprocessing.StandardScaler().fit(features)
            self.features = self.scaler.transform(features)
        else:
            self.features = features

        if self.k_neigs >= l_train:
            self.k_neigs = l_train - 1
            # print("Max k reached!")
        # print(f"{self.k_neigs} and {l_train}")
        if self.adaptative:
            adaptive_knn = AdaptiveKNN(min_k=1, max_k=self.k_neigs)
            G, edges_dest, edges_source, weights, data = adaptive_knn.fit(
                self.features, y_true_numeric
            )
            # G, edges_dest, edges_source, weights, data = adaptive_knn.predict(self.features)
            self.G = G
            # data = from_networkx(self.G)
            self.weights = weights
        else:
            self.Adj = kneighbors_graph(
                self.features, self.k_neigs, metric="cosine", mode=self.mode
            )
            self.A = self.operar_matriz_esparsa(self.Adj.copy(), self.k_neigs)
            # self.G = nx.Graph()
            weights = []
            edges_source = []
            edges_dest = []

            cx = self.A.tocoo()
            for i, j, v in zip(cx.row, cx.col, cx.data):
                if v > 0.01:
                    edges_source.append(i)
                    edges_source.append(j)
                    edges_dest.append(j)
                    edges_dest.append(i)
                    weights.append([v])
                    weights.append([v])
            self.weights = np.array(weights)
        data = Data(
            # Atributos dos nós
            x=torch.tensor(features, dtype=torch.float),
            # Conexões das arestas
            edge_index=torch.tensor([edges_source, edges_dest], dtype=torch.long),
            # Atributos das arestas
            weights=torch.tensor(self.weights, dtype=torch.float),
            y=torch.tensor(y_true_numeric, dtype=torch.long),
        )
        # Add additional arguments to `data`:
        data.train_idx = torch.tensor(np.arange(l_train), dtype=torch.long)

        a = torch.zeros(len(data.y), dtype=torch.bool)
        a[data.train_idx] = True
        data.train_mask = a

        if X_val is not None:
            data.val_idx = torch.tensor(
                np.arange(l_train, l_val + l_train), dtype=torch.long
            )
            a = torch.zeros(len(data.y), dtype=torch.bool)
            a[data.val_idx] = True
            data.val_mask = a

        if X_test is not None:
            l_test = len(X_test)
            data.test_idx = torch.tensor(
                np.arange(l_val + l_train, l_val + l_train + l_test), dtype=torch.long
            )
            a = torch.zeros(len(data.y), dtype=torch.bool)
            a[data.test_idx] = True
            data.test_mask = a

        # Analyzing the graph structure:
        if self.verbose:
            print(type(features), features.shape)
            print("Shape of features for all nodes:", features.shape)
            print("Shape scaled of all feats:", self.features.shape)
            print("Total nodes:", data.num_nodes)

            print("Graph is directed?", data.is_directed())
            print("Shape of data.x:", data.x.shape[1])
            print("Classes in graph:", np.unique(data.y))
            print("Graph Networkx:", self.G)

        return data

    def plot_graph(self):
        if self.G is not None:

            G = self.G
            pos = nx.spring_layout(
                G, seed=7
            )  # positions for all nodes - seed for reproducibility
            node_color = [node[1]["color"] for node in G.nodes(data=True)]
            node_name = [
                f"i{i}:c{int(node[1]['name'])}"
                for i, node in enumerate(G.nodes(data=True))
            ]
            # nodes
            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=700,
                nodelist=G.nodes(),
                node_color=node_color,
                label=node_name,
            )

            elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
            esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

            # edges
            nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=esmall,
                width=6,
                alpha=0.5,
                edge_color="b",
                style="dashed",
            )
            # pos = {n: pos[k] for k, n in zip(pos.keys(), node_name)}
            node_name = {i: v for i, v in enumerate(node_name)}
            # node labels
            nx.draw_networkx_labels(
                G, pos, labels=node_name, font_size=15, font_family="sans-serif"
            )
            # edge weight labels
            edge_labels = nx.get_edge_attributes(G, "weight")
            edge_labels = {k: round(v, 2) for k, v in edge_labels.items()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels)

            ax = plt.gca()
            ax.margins(0.08)
            plt.axis("off")
            plt.tight_layout()
            plt.show()
