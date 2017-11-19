#!/usr/bin/env python
# encoding: utf-8

"""
Automated tests for paper2vec
"""

import unittest
import os

datapath_cites = os.path.join(os.path.dirname(__file__), "test_data", "cora.cites")
datapath_content = os.path.join(os.path.dirname(__file__), "test_data", "cora.content")

from gensim import utils
from gensim.models import paper2vec

class TestPaper2VecModel(unittest.TestCase):

    def test_paper2vec_init(self):
        p2v = paper2vec.Paper2Vec(papers_file = datapath_content, citation_graph_file = datapath_cites)

    def test_load_data(self):
        p2v = paper2vec.Paper2Vec()
        p2v.load_data(papers_file = datapath_content, citation_graph_file = datapath_cites)

    def test_train(self):
        d2v_dict = {'alpha': 0.025, 'window': 10, 'min_count': 10, 'min_alpha': 0.025, 'size': 100}
        w2v_dict = {'size': 100, 'window': 5}
        p2v = paper2vec.Paper2Vec(papers_file = datapath_content, citation_graph_file = datapath_cites,
        d2v_dict = d2v_dict, w2v_dict = w2v_dict)
        p2v.train()

    def test_predict(self):
        pass

if __name__ == "__main__":
    unittest.main()