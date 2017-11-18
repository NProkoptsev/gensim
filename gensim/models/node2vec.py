"""
Produce node vectors with deep learning via DeepWalk algorithm or Node2Vec algorithm
[1] Bryan Perozzi, Rami Al-Rfou, Steven Skiena. DeepWalk: Online Learning of Social Representations
https://arxiv.org/abs/1403.6652
[2] Aditya Grover, Jure Leskovec. node2vec: Scalable Feature Learning for Networks
https://arxiv.org/pdf/1607.00653.pdf
"""

from abc import ABC, abstractmethod
import random
from collections import defaultdict
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
import numpy as np
try:
    from gensim.models.word2vec_inner import MAX_WORDS_IN_BATCH
except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    MAX_WORDS_IN_BATCH = 10000


class GraphRandomWalk():
    """
    Class for creating random walks on graph
    Args:
      data : tuple of Adjacency list of graph, frequencies of nodes
    """

    def __init__(self, data):
        self.adj_list = data[0]
        self.vertex_frequencies = data[1]

    @classmethod
    def from_filename(cls, filename):
        """
        Instantiate class from file
        Args:
          filename : path to file with edge list
        """
        adj_list = defaultdict(list)
        frequencies = defaultdict(int)
        with open(filename, 'r') as file:
            edges = [tuple(map(int, line.split())) for line in file]
            for edge in edges:
                adj_list[edge[0]].append(edge[1])
                frequencies[edge[1]] += 1
        return cls((adj_list, frequencies))

    @classmethod
    def from_edgelist(cls, edgelist):
        """
        Instantiate class from list of edges
        Args:
          edgelist : list of edges
        """
        adj_list = defaultdict(list)
        frequencies = defaultdict(int)
        for edge in edgelist:
            adj_list[edge[0]].append(edge[1])
            frequencies[edge[1]] += 1
        return cls((adj_list, frequencies))

    @property
    def vertices_count(self):
        """
        Get number of vertices in graph
        """
        return len(self.adj_list)

    @property
    def frequencies(self):
        """
        Get vertex frequencies
        Returns:
          List of frequencees of size vertices_count
        """
        return self.vertex_frequencies

    def add_edge(self, edge):
        """
        Adds edge to graph
        Args:
          edge : tuple of 2 vertex
        """
        self.adj_list[edge[0]].append(edge[1])
        self.vertex_frequencies[edge[1]] += 1

    def adj(self, vertex):
        """
        Get adjacent vertices of some vertex
        Args:
          vertex : vertex number
        Returns
          List of vertices
        """
        return self.adj_list.get(vertex, [])

    def degree(self, vertex):
        """
        Get degree of vertex
        Args:
          vertex : vertex number
        """
        return len(self.adj_list.get(vertex, []))

    def random_walk(self, vertex, length):
        """
        Make random walk on graph
        Args:
          vertex : vertex where random walk start from
          length : length of randow walk
        Returns:
          List of vertices
        """
        sequence = [vertex]
        for _ in range(length - 1):
            adj = self.adj(vertex)
            vertex = adj[random.uniform(0, self.degree(vertex))]
            sequence.append(vertex)
        return sequence

    def bulk_random_walk(self, length, bulk_size):
        """
        Makes a list of random walks
        Args:
          length : length of randow walk
          bulk_size : number of random walks per node
        Returns:
          List of random walks, where random walk is a list of vertices
        """
        sequnece = []
        for _ in range(bulk_size):
            for j in range(self.vertices_count):
                sequnece.append(self.random_walk(j, length))
        return sequnece


class GraphBiasedWalk(GraphRandomWalk):
    """
    Class for making biased random walks on graph described
    Args:
    graph : instance of GraphRandomWalk
    p : return parametr
    q : in-out parametr
    """

    def __init__(self, graph, p, q):
        if not isinstance(graph, GraphRandomWalk):
            raise Exception('Not correct graph')
        self.graph = graph
        self.probs = self.second_order_adj_list(p, q)

    def second_order_adj_list(self, p, q):
        """
        Make a second order list of random walk probabilities for all nodes
        Args:
          p : return parametr
          q : in-out parametr
        Retuns:
          List of random walk probabilities of graph. One list for each node
        """
        probs = {}

        def calculate_prob_dict(prev_node):
            """
            Make a second order list of random walk probabilities for one node
            Args:
              prev_node : previous node in random walk  
            """
            new_dict = {}
            prev_adj = self.graph.adj_list[prev_node]
            for node, cur_adj in self.graph.adj_list:
                new_list = []
                for adj_node in cur_adj:
                    if adj_node == prev_node:
                        prob = 1 / p
                    else:
                        if adj_node in prev_adj:
                            prob = 1
                        else:
                            prob = 1 / q
                    new_list.append(prob)
                new_dict[node] = new_list
            return new_dict

        for prev_node in self.graph.adj_list.keys():
            probs[prev_node] = calculate_prob_dict(prev_node)
        return probs

    @property
    def vertices_count(self):
        return self.graph.vertices_count

    @property
    def frequencies(self):
        return self.graph.vertex_frequencies

    def add_edge(self, edge):
        raise Exception("Reinitialize this class with new graph instead")

    def adj(self, vertex):
        return self.graph.adj(vertex)

    def degree(self, vertex):
        return self.graph.degree(vertex)

    def random_walk(self, vertex, length):
        """
        Makes a list of biased random walks
        Args:
          length : length of randow walk
          bulk_size : number of random walks per node
        Returns:
          List of random walks, where random walk is a list of vertices
        """
        sequence = [vertex]
        prev_vertex = vertex
        current_vertex = self.adj(
            vertex)[random.uniform(0, self.degree(vertex))]
        sequence.append(current_vertex)
        for _ in range(length - 2):
            adj = self.adj(vertex)
            prob = self.probs[prev_vertex][current_vertex]
            prob = np.array(prob) / sum(prob)
            prev_vertex = current_vertex
            current_vertex = np.random.choice(adj, p=prob)
            sequence.append(vertex)
        return sequence

    def bulk_random_walk(self, length, bulk_size):
        sequnece = []
        for _ in range(bulk_size):
            for j in range(self.vertices_count):
                sequnece.append(self.random_walk(j, length))
        return sequnece


class Node2Vec(Word2Vec):
    """
    Class for training, using and evaluating neural networks described in https://arxiv.org/pdf/1607.00653.pdf
    Args:
      graph : instance of GraphRandomWalk
      rw_length : length of random walk 
      bulk_size : number of random walks per node
    """

    def __init__(self, graph=None, rw_length=40, bulk_size=10, size=100, alpha=0.025, window=5, min_count=True,
                 sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False):
        if (rw_length < 2):
            raise Exception("Length can't be less than 2")
        self.rw_length = rw_length
        self.bulk_size = bulk_size
        super(Node2Vec, self).__init__(None, size, alpha, window, 0,
                                       sample, seed, workers, min_alpha,
                                       sg, hs, negative, cbow_mean, hashfxn, iter, null_word,
                                       None, sorted_vocab, batch_words, compute_loss)

        if graph != None:
            self.build_vocab(graph)
            self.train(graph, total_examples=self.corpus_count,
                       epochs=self.iter, start_alpha=self.alpha, end_alpha=self.min_alpha)

    def build_vocab(self, graph, keep_raw_vocab=False, update=False):
        """
        Build vocabulary from frequencies of vertices in random walks, which we approximate with in degree of vertex
        Args:
          graph : instance of GraphRandomWalk
        """
        frequencies = graph.frequencies
        corpus_count = self.bulk_size * graph.vertices_count
        super(Node2Vec, self).build_vocab_from_freq(
            frequencies, keep_raw_vocab, corpus_count, None, update)

    def build_vocab_from_freq(self, word_freq, keep_raw_vocab=False, corpus_count=None, trim_rule=None, update=False):
        raise Exception('Not supported, use build_vocab() instead')

    def train(self, graph, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None, word_count=0,
              queue_factor=2, report_delay=1.0, compute_loss=None):
        """
        Update the model's neural weights from a list of random walks
        Args:
          graph : instance of GraphRandomWalk
        """
        sentences = graph.bulk_random_walk(self.rw_length, self.bulk_size)
        super(Node2Vec, self).train(sentences, total_examples=self.corpus_count,
                                    epochs=self.iter, start_alpha=self.alpha, end_alpha=self.min_alpha)
