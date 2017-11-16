from abc import ABC, abstractmethod
import logging
import os
import warnings
import random
from gensim import utils, matutils
# , train_cbow_pair, train_sg_pair, train_batch_sg
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.keyedvectors import KeyedVectors, Vocab
try:
    from gensim.models.word2vec_inner import MAX_WORDS_IN_BATCH
except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    MAX_WORDS_IN_BATCH = 10000


class Graph(ABC):

    @abstractmethod
    @classmethod
    def from_filename(cls, filename):
        pass

    @abstractmethod
    @classmethod
    def from_list(cls, lst):
        pass

    @abstractmethod
    @property
    def vertices_count(self):
        pass

    @abstractmethod
    def add_edge(self, edge):
        pass

    @abstractmethod
    def adj(self, vertex):
        pass

    @abstractmethod
    def degree(self, vertex):
        pass


class GraphListImpl(Graph):

    def __init__(self, data):
        self.adj_list = data

    @classmethod
    def from_filename(cls, filename):
        adj_list = {}
        with open(filename, 'r') as file:
            edges = [tuple(map(int, line.split())) for line in file]
            for edge in edges:
                adj_list.setdefault(edge[0], []).append(edge[1])
                adj_list.setdefault(edge[1], []).append(edge[0])
        return cls(adj_list)

    @classmethod
    def from_list(cls, lst):
        return cls(lst)

    @property
    def vertices_count(self):
        return len(self.adj_list)

    def add_edge(self, edge):
        self.adj_list.setdefault(edge[0], []).append(edge[1])
        self.adj_list.setdefault(edge[1], []).append(edge[0])

    def adj(self, vertex):
        return self.adj_list.get(vertex, [])

    def degree(self, vertex):
        return len(self.adj_list.get(vertex, []))

    def random_walk(self, vertex, length):
        sequence = [vertex]
        for _ in range(length):
            adj = self.adj(vertex)
            vertex = adj[random.uniform(0, self.degree(vertex))]
            sequence.append(vertex)
        return sequence

    def bulk_random_walk(self, length, bulk_size):
        sequnece = []
        for i in range(bulk_size):
            for j in range(self.vertices_count):
                sequnece.append(self.random_walk(j, length))
        return sequnece

    def get_biased_sequence(self, length):
        pass


class Node2Vec(Word2Vec):
    def __init__(self, graph=None, rw_length=10, bulk_size=40, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False):
        super(Node2Vec, self).__init__(None, size, alpha, window, min_count,
                                       max_vocab_size, sample, seed, workers, min_alpha,
                                       sg, hs, negative, cbow_mean, hashfxn, iter, null_word,
                                       trim_rule, sorted_vocab, batch_words, compute_loss)
        self.rw_length = rw_length
        self.bulk_size = bulk_size
        if graph != None:
            if not isinstance(graph, Graph):
                raise Exception('Not correct graph')
            sentences = graph.bulk_random_walk(self.rw_length, self.bulk_size)
            self.build_vocab(sentences, trim_rule=trim_rule)
            super(Node2Vec, self).train(sentences, total_examples=self.corpus_count,
                                        epochs=self.iter, start_alpha=self.alpha, end_alpha=self.min_alpha)

    def train(self, graph, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None, word_count=0,
              queue_factor=2, report_delay=1.0, compute_loss=None):
        if not isinstance(graph, Graph):
            raise Exception('Not correct graph')
        sentences = graph.bulk_random_walk(self.rw_length, self.bulk_size)
        super(Node2Vec, self).train(sentences, total_examples=self.corpus_count,
                                    epochs=self.iter, start_alpha=self.alpha, end_alpha=self.min_alpha)


class Paper2Vec():
    def __init__(self, doc2vec_model, node2vec_model):
        pass

    def train(self, citations_filename, bow_filenmae):
        '''1. Use doc2vec
           2. Init graph
           3. Add new edges to graph
           4. Use Node2Vec '''
        pass
