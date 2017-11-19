""" Paper2Vec
Encode papers into vectors!
Implementation of the paper: https://arxiv.org/abs/1703.06587
"""
from __future__ import division, print_function
from abc import ABC, abstractmethod
import random
from collections import defaultdict, namedtuple, Sequence
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.node2vec import GraphRandomWalk, Node2Vec
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
        reduce_memory: A boolean variable, set True will make to try hard
            to reduce memory and delete unused data structures.
        topn: An integer, number of neighbours in Doc2Vec to be added to
            citation graph.
    """

    def __init__(self, papers=None, citation_graph=None, papers_file=None,
                 citation_graph_file=None, d2v_dict=None, w2v_dict=None, reduce_alpha=False,
                 seed=None, reduce_memory=False, topn=2,
                 **kwargs):

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
        """Set file names or data structures for papers and citations.

        DO NOT populate your memory with data, hence, use it if you
        want to change data processed.
        Files have higher priority and will substutute given datastructures.
        For example, when both `papers` and `papers_file` are set, the
        information will be taken from `papers_file`.

        Args:
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
        Returns:
            None
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


    def train(self):
        """Start memory population with data and train models.

        Args:
            None
        Returns:
            None
        """
        # Populate environment with parced data
        if self.__papers_as_file is not None:
            self.__papers.papers = self.__papers_as_file
        else:
            self.__papers.papers = self.__papers_as_list

        # Init citation graph
        if self.__citation_graph_as_file is not None:
            self.__graph = GraphRandomWalk.from_filename(self.__citation_graph_as_file)
        else:
            self.__graph = GraphRandomWalk.from_edgelist(self.__citation_graph_as_list)

        # Build Doc2Vec
        model_d2v = Doc2Vec(**self.__d2v_dict)
        model_d2v.build_vocab(self.__papers.papers)
        if self.__seed is not None:
            random.seed(self.__seed)

        # Reduce alpha
        if self.__reduce_alpha:
            for i in range(10):
                self.__papers.shuffle()
                model_d2v.alpha = 0.025-0.002*i
                model_d2v.min_alpha = model_d2v.alpha
                model_d2v.train(self.__papers.papers)

        # Add similar from d2v edges to graph
        for paper in self.__papers.papers:
            for node in model_d2v.docvecs.most_similar(paper.tags,topn=self.__topn):
                edge = (paper.tags[0], node[0])
                self.__graph.add_edge(edge)

        # Final steps. Node2Vec
        self.__paper2vec = Node2Vec(graph=self.__graph, **self.__w2v_dict)

    def __getitem__(self, index):
        if isinstance(index, [str, int]):
            return self.__paper2vec[index]
        else:
            raise TypeError('index must be string or integer!')

class _Papers(object):
    """A class for papers encapsulation.

    Attributes:
        papers: A datastructure (list of namen tuples, see Paper2Vec) with papers
        papers_file: A string with papers file name (see Paper2Vec)
    """
    def __init__(self, papers=None, papers_file=None):
        if papers is not None:
            self.__papers = papers
        elif papers_file is not None:
            self.__papers = self.__parse_papers_file(papers_file)

    def __parse_papers_file(self, papers_file):
        """Processe `papers_file` into `papers`
        Idea of code from https://github.com/asxzy/paper2vec-gensim/blob/master/gensim.ipynb

        Args:
            papers_file: A string with papers file name (see Paper2Vec)

        Returns:
            dataset: A datastructure (list of namen tuples, see Paper2Vec) with papers
        """
        dataset = []
        paper = namedtuple('paper', 'words tags')
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
        """Getter for papers

        Args:
            None

        Returns:
            __papers: A list of named tuples
        """
        return self.__papers

    @papers.setter
    def papers(self, papers_var):
        """Setter for papers.

        Args:
            papers_var: A string for file name or a list of named tuples

        Returns:
            None
        """
        if isinstance(papers_var, str):
            self.__papers = self.__parse_papers_file(papers_var)
        else:
            self.__papers = papers_var.copy()

    def shuffle(self):
        """Shuffles papers dataset for alpha

        Args:
            None

        Returns:
            None
        """
        random.shuffle(self.__papers)


"""Exception when user did not provide full data"""
class MissingData(Exception):
    pass
