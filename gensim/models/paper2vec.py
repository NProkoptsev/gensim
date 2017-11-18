""" Paper2Vec"""
from __future__ import division, print_function
from abc import ABC, abstractmethod
import random
from collections import defaultdict, namedtuple, Sequence
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from node2vec import GraphDeepWalk, Node2Vec
try:
    from gensim.models.word2vec_inner import MAX_WORDS_IN_BATCH
except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    MAX_WORDS_IN_BATCH = 10000


class Paper2Vec(object):
    """Class for training paper graph.

    It combines Doc2Vec and Word2Vec with Node2Vec to produce mapping
    for papers into vectors. This class becomes a dict and takes integers or
    strings as indexes.
    Typical usage is:
    * build dictionaries with parameters for Word2Vec and Doc2Vec.
        See documentation for W2V and D2V and Paper2Vec's paper:
        https://arxiv.org/abs/1703.06587
    * make an instance and provide papers and citation_graph structures or
        file names: model = Paper2Vec(file, cit_file, blah)
    * use model.load_data(blah) to change links of data. If you do not change
        data, it's a redundant action. But you can change data not
        re-initializing the class.
    * whether you changed data or not, call model.train(blah). train() takes
        all parameters which are taken by initialization except links to data.

    Attributes:
        papers: Papers represented as
            [('words'=['word or num of word from BOW', ...], 'tags'=[ID]), # namedtuple
                ...
            )]
        citation_graph: A list if tuples (edges) such as [(ID1, ID2), ...]
        papers_file: A text file with the format
            ID1 bag_of_words tag
            ...
        citation_graph_file: A text file where there is an edge on every line
            just like ID1[space]ID2.
        d2v_dict: A dictionary with parameters for Doc2Vec.
        w2v_dict: A dictionary with parameters for Word2Vec.
        reduce_alpha: A boolean variable for retraining model or not with
            reducing alpha
        seed: An integer which will be set to random initialization
    """

    def __init__(self, papers=None, citation_graph=None, papers_file=None,
                 citation_graph_file=None, d2v_dict=None, w2v_dict=None, reduce_alpha=False,
                 seed=None, reduce_memory=False, topn=2,
                 **kwargs):
        """
        `papers` is

        `alpha` is the initial learning rate (will linearly drop to `min_alpha`
            as training progresses)

        `min_alpha` is minimum learning rate

        `citation_graph` is the list of tuples of 2 IDs, which are actually
            edges. What is if paper with ID 1234 cites paper with ID 4321, then
            [(1234, 4321), ...]

        `papers_file` is a text file with lines as foolowing:
            ID_of_paper bag_of_words tag

        `citation_graph`

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
