""" Paper2Vec"""

from abc import ABC, abstractmethod
import random
from collections import defaultdict
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
try:
    from gensim.models.word2vec_inner import MAX_WORDS_IN_BATCH
except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    MAX_WORDS_IN_BATCH = 10000


class Graph(ABC):
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

    @abstractmethod
    def bulk_random_walk(self, length, bulk_size):
        pass


class GraphDeepWalk(Graph):

    def __init__(self, filename):
        self.adj_list = defaultdict(list)
        self.frequences = defaultdict(int)
        with open(filename, 'r') as file:
            edges = [tuple(map(int, line.split())) for line in file]
            for edge in edges:
                self.adj_list[edge[0]].append(edge[1])
                self.frequences[edge[1]] += 1
                # self.adj_list.setdefault(edge[1], []).append(edge[0])

    @property
    def vertices_count(self):
        return len(self.adj_list)

    def add_edge(self, edge):
        self.adj_list[edge[0]].append(edge[1])
        self.frequences[edge[1]] += 1
        # self.adj_list.setdefault(edge[1], []).append(edge[0])

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
        for _ in range(bulk_size):
            for j in range(self.vertices_count):
                sequnece.append(self.random_walk(j, length))
        return sequnece


class Node2Vec(Word2Vec):
    def __init__(self, graph=None, rw_length=40, bulk_size=10, size=100, alpha=0.025, window=5, min_count=True,
                 sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False):
        self.rw_length = rw_length
        self.bulk_size = bulk_size
        super(Node2Vec, self).__init__(None, size, alpha, window, 0,
                                       sample, seed, workers, min_alpha,
                                       sg, hs, negative, cbow_mean, hashfxn, iter, null_word,
                                       None, sorted_vocab, batch_words, compute_loss)

        if graph != None:
            if not isinstance(graph, Graph):
                raise Exception('Not correct graph')
            self.build_vocab(graph)
            self.train(graph, total_examples=self.corpus_count,
                       epochs=self.iter, start_alpha=self.alpha, end_alpha=self.min_alpha)

    def build_vocab(self, graph, keep_raw_vocab=False, update=False):
        frequences = graph.frequences
        corpus_count = self.bulk_size * graph.vertices_count
        super(Node2Vec, self).build_vocab_from_freq(
            frequences, keep_raw_vocab, corpus_count, None, update)

    def build_vocab_from_freq(self, word_freq, keep_raw_vocab=False, corpus_count=None, trim_rule=None, update=False):
        raise Exception('Not supported, use build_vocab() instead')

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
