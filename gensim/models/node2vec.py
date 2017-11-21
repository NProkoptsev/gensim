"""
Produce node vectors with deep learning via DeepWalk algorithm or Node2Vec algorithm
[1] Bryan Perozzi, Rami Al-Rfou, Steven Skiena. DeepWalk: Online Learning of Social Representations
https://arxiv.org/abs/1403.6652
[2] Aditya Grover, Jure Leskovec. node2vec: Scalable Feature Learning for Networks
https://arxiv.org/pdf/1607.00653.pdf
"""

from collections import defaultdict
from gensim.models.word2vec import Word2Vec
import numpy as np

import logging
logger = logging.getLogger(__name__)


class RandomWalkFactory():
    """
    Class for creating random walks on graph
    Args:
      data : tuple of Adjacency list of graph, frequencies of nodes
    """

    def __init__(self, data):
        self.adj_list = data

    @classmethod
    def from_filename(cls, filename):
        """
        Instantiate class from file
        Args:
          filename : path to file with edge list
        """
        adj_list = defaultdict(set)
        with open(filename, 'r') as file:
            edges = [tuple(map(int, line.split())) for line in file]
            for edge in edges:
                adj_list[edge[0]].add(edge[1])
                adj_list[edge[1]].add(edge[0])
        return cls(adj_list)

    @classmethod
    def from_edgelist(cls, edgelist):
        """
        Instantiate class from list of edges
        Args:
          edgelist : list of edges
        """
        adj_list = defaultdict(set)
        for edge in edgelist:
            adj_list[edge[0]].add(edge[1])
            adj_list[edge[1]].add(edge[0])
        return cls(adj_list)

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
        return {str(vertex): self.degree(vertex) for vertex in self.adj_list.keys()}

    def add_edge(self, edge):
        """
        Adds edge to graph
        Args:
          edge : tuple of 2 vertex
        """
        self.adj_list[edge[0]].add(edge[1])
        self.adj_list[edge[1]].add(edge[0])

    def adj(self, vertex):
        """
        Get adjacent vertices of some vertex
        Args:
          vertex : vertex number
        Returns
          List of vertices
        """
        return tuple(self.adj_list[vertex])

    def degree(self, vertex):
        """
        Get degree of vertex
        Args:
          vertex : vertex number
        """
        return len(self.adj_list[vertex])

    def random_walk(self, vertex, length):
        """
        Make random walk on graph
        Args:
          vertex : vertex where random walk start from
          length : length of randow walk
        Returns:
          List of vertices
        """
        sequence = [str(vertex)]
        for _ in range(length - 1):
            vertex = np.random.choice(self.adj(vertex))
            sequence.append(str(vertex))
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
            for vertex in self.adj_list.keys():
                sequnece.append(self.random_walk(vertex, length))
        return sequnece


class BiasedRandomWalkFactory(RandomWalkFactory):
    """
    Class for making biased random walks on graph described
    Args:
    graph : instance of RandomWalkFactory
    p : return parametr
    q : in-out parametr
    """

    def __init__(self, graph, p, q):
        if not isinstance(graph, RandomWalkFactory):
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
            for node, cur_adj in self.graph.adj_list.items():
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
        sequence = [str(vertex)]
        prev_vertex = vertex
        current_vertex = np.random.choice(self.adj(vertex))
        sequence.append(str(current_vertex))
        for _ in range(length - 2):
            adj = self.adj(current_vertex)
            prob = self.probs[prev_vertex][current_vertex]
            prob = np.array(prob) / sum(prob)
            prev_vertex = current_vertex
            current_vertex = np.random.choice(adj, p=prob)
            sequence.append(str(current_vertex))
        return sequence

    def bulk_random_walk(self, length, bulk_size):
        sequnece = []
        for _ in range(bulk_size):
            for vertex in self.graph.adj_list.keys():
                sequnece.append(self.random_walk(vertex, length))
        np.random.shuffle(sequnece)
        return sequnece


class Node2Vec(Word2Vec):
    """
    Class for training, using and evaluating neural networks described in https://arxiv.org/pdf/1607.00653.pdf
    Args:
      graph : instance of RandomWalkFactory
      rw_length : length of random walk 
      bulk_size : number of random walks per node
    """

    def __init__(self, graph=None, rw_length=40, bulk_size=10, **kwargs):
        if (rw_length < 2):
            raise Exception("Length can't be less than 2")
        self.rw_length = rw_length
        self.bulk_size = bulk_size
        super(Node2Vec, self).__init__(sentences=None,
                                       sorted_vocab=0,  min_count=0, **kwargs)

        if graph != None:
            self.build_vocab(graph)
            self.train(graph, epochs=self.iter,
                       start_alpha=self.alpha, end_alpha=self.min_alpha)

    def build_vocab(self, graph, **kwargs):
        """
        Build vocabulary from frequencies of vertices in random walks, which we approximate with in degree of vertex
        Args:
          graph : instance of RandomWalkFactory
        """
        self.sentences = graph.bulk_random_walk(self.rw_length, self.bulk_size)
        super(Node2Vec, self).build_vocab(self.sentences, trim_rule=None, **kwargs )

    def build_vocab_from_freq(self, graph, keep_raw_vocab=False, update=False):
        """
        Build vocabulary from frequencies of vertices in random walks, which we approximate with in degree of vertex
        Args:
          graph : instance of RandomWalkFactory
        """
        frequencies = graph.frequencies
        corpus_count = self.bulk_size * graph.vertices_count
        super(Node2Vec, self).build_vocab_from_freq(
            frequencies, keep_raw_vocab, corpus_count, None, update)

    def train(self, graph, **kwargs):
        """
        Update the model's neural weights from a list of random walks
        Args:
          graph : instance of RandomWalkFactory
        """
        if self.sentences is None:
            self.sentences = graph.bulk_random_walk(self.rw_length, self.bulk_size)
        super(Node2Vec, self).train(
            self.sentences, total_examples=self.corpus_count, **kwargs)
