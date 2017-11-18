""" Paper2Vec"""
from __future__ import division, print_function
from abc import ABC, abstractmethod
import random
from collections import defaultdict, namedtuple, Sequence
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

class Paper2Vec:
    """
    Class for training paper graph and mapping input papers to some vector
    representation
    """

    def __init__(self, papers=None, citation_graph=None, papers_file=None,
                 citation_graph_file=None, d2v_dict=None, w2v_dict=None, reduce_alpha=False,
                 seed=None, reduce_memory=False, topn=2,
                 **kwargs):
        """
        `papers` is papers represented as (Name of DB what ever you like)
            [('Name of Database'=['Num of word', ...], 'word tags'=[ID]), # namedtuple
                ...
            )]

        `alpha` is the initial learning rate (will linearly drop to `min_alpha`
            as training progresses)

        `min_alpha` is minimum learning rate

        `citation_graph` is the list of tuples of 2 IDs, which are actually
            edges. What is if paper with ID 1234 cites paper with ID 4321, then
            [(1234, 4321), ...]

        `papers_file` is a text file with lines as foolowing:
            ID_of_paper bag_of_words tag

        `citation_graph` list if tuples (edges) such as [(ID1, ID2), ...]

        `window` **TODO**

        `citation_graph_file` is a text file where every edge in
            ID1 ID2
            format (see cicitation_graph) is represented on every line

        `seed` is set for reproducing random activities.
            Will be set before every random generator

        `topn` amount of most similar papers to be added from Doc2Vec
        `workers` is amount of threads for training the model
        """
        self.__reduce_memory = reduce_memory
        self.__d2v_dict = d2v_dict
        self.__w2v_dict = w2v_dict
        self.__seed = seed
        self.__reduce_alpha = reduce_alpha
        self.__topn = topn

        self.__papers_as_list = papers
        self.__papers_as_file = papers_file
        self.__papers = _Papers()
        self.__citation_graph_as_list = citation_graph
        self.__citation_graph_as_file = citation_graph_file


    def load_data(self, papers=None, citation_graph=None, papers_file=None,
                  citation_graph_file=None):
        """
        Will rewrite already set data, if it was set in initialization.

        DO NOT populate your memory with data, hence, use it if you
        want to change data processed.

        Files have higher priority and will substutute given datastructures.
        For example, when both `papers` and `papers_file` are set, the
        information will be taken from `papers_file`.
        """

        if papers_file is not None:
            if self.__reduce_memory:
                self.__papers_as_list = None
            self.__papers_as_file = papers_file
        elif papers is not None:
            if self.__reduce_memory:
                self.__papers_as_file = None
            self.__papers_as_list = papers

        if citation_graph_file is not None:
            if self.__reduce_memory:
                self.__citation_graph_as_list = None
            self.__citation_graph_as_file = citation_graph_file
        elif citation_graph is not None:
            if self.__reduce_memory:
                self.citation_graph_file = None
            self.__citation_graph_as_list = citation_graph

        self.__paper2vec = dict()

    def train(self, d2v_dict=self.__d2v_dict, w2v_dict=self.__w2v_dict,
              seed=self.__seed, reduce_alpha=self.__reduce_alpha,
              topn=self.__topn):
        """
        Start memory population with data and train models
        """
        # Populate environment with parced data
        if self.__papers_as_file is not None:
            self.__papers.papers = self.__papers_as_file
        else:
            self.__papers.papers = self.__papers_as_list

        # Init citation graph
        if self.__citation_graph_file is not None:
            self.__graph = GraphDeepWalk.from_file(self.__citation_graph_as_file)
        else:
            self.__graph = GraphDeepWalk.from_egdelist(self.__citation_graph_as_list)

        # Build Doc2Vec
        model_d2v = Doc2Vec(**d2v_dict)
        model_d2v.build_vocab(self.__papers.papers)
        if seed is not None:
            random.set_seed(seed)

        # Reduce alpha
        if reduce_alpha:
            for i in range(10):
                self.__papers.shuffle()
                model_d2v.alpha = 0.025-0.002*i
                model_d2v.min_alpha = model_d2v.alpha
                model_d2v.train(self.__papers.papers)

        # Add similar from d2v edges to graph
        for paper in self.__papers.papers:
            for node in model.docvecs.most_similar(paper.ID,topn=topn):
                edge = (paper.ID[0], node[0])
                self.__graph.add_edge(edge)

        # Final steps. Node2Vec
        self.__paper2vec = Node2Vec(graph=self.__graph, **w2v_dict)

    def __getitem__(self, index):
        if isinstance(index, string_types + integer_types + (integer,)):
            return self.__paper2vec[index]
        else:
            raise TypeError('index must be string or integer!')


class _Papers:
    """
    Make class for papers and labels encapsulation
    """
    def __init__(self):
        """
        `papers` is papers in described at Paper2Vec format
        `papers_file` is pappers file in described at Paper2Vec format
        """
        if papers is not None:
            self.__papers = papers
        elif papers_file is not None:
            self.__papers = __parse_papers_file(papers_file)

    def __parse_papers_file(self, papers_file):
        """
        Processe `papers_file` into `papers`
        Idea of code from https://github.com/asxzy/paper2vec-gensim/blob/master/gensim.ipynb
        """
        dataset = []
        paper = namedtuple('paper', 'words ID')
        with open(papers_file) as f:
            for line in f:
                line = line.split()
                ID = line[0]
                words = []
                for i, w in enumerate(line):
                    if w == "1":
                        words.append(str(i))
                dataset.append(paper(words, [ID]))
        return dataset

    @property
    def papers(self):
        return self.__papers

    @papers.setter
    def papers(self, papers_var):
        if isinstance(papers_var, str):
            self.__papers = __parse_papers_file(self, papers_var)
        else:
            self.__papers = papers_var.copy()

    def shuffle():
        random.shuffle(self.__papers)

class MissingData(Exception):
    pass
