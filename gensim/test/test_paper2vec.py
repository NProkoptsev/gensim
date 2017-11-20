#!/usr/bin/env python
# encoding: utf-8

"""
Automated tests for paper2vec
"""

import unittest
import os

from gensim import utils
from gensim.models import paper2vec
from collections import namedtuple

datapath_cites = os.path.join(os.path.dirname(__file__), "test_data", "cora.cites")
datapath_content = os.path.join(os.path.dirname(__file__), "test_data", "cora.content")

paper = namedtuple('paper', 'words tags')

# Test dataset
contents_test = [paper(words=['119', '126', '177', '253', '352', '457', '508', '522', '620', '649', '699', '703', '735', '846', '903', '1206', '1210', '1237', '1353', '1427'], tags=['31336']),
    paper(words=['13', '510', '621', '764', '883', '894', '979', '1132', '1136', '1178', '1208', '1257', '1264', '1267', '1333', '1390', '1426'], tags=['1061127']),
    paper(words=['46', '210', '213', '240', '293', '395', '511', '515', '582', '622', '624', '639', '1076', '1133', '1178', '1207', '1264', '1290', '1350', '1390', '1416', '1422'], tags=['1106406']),
    paper(words=['42', '94', '100', '150', '595', '618', '625', '649', '875', '916', '943', '989', '1005', '1050', '1072', '1171', '1178', '1195', '1293', '1349', '1350'], tags=['13195']),
    paper(words=['45', '123', '136', '154', '365', '397', '403', '475', '508', '620', '662', '700', '702', '829', '1067', '1175', '1176', '1178', '1209', '1210', '1213', '1255', '1382'], tags=['37879']),
    paper(words=['94', '169', '212', '508', '527', '552', '875', '973', '1144', '1178', '1199', '1291', '1427'], tags=['1126012'])]
citation_test = [(31336, 1126012), (31336, 1061127), (1061127, 13195), (1061127, 37879), (1106406, 1126012)]

# Test parameters
d2v_params = {'alpha': 0.025, 'window': 10, 'min_count': 10, 'min_alpha': 0.025, 'size': 100}
w2v_params = {'size': 100, 'window': 5}

class TestPaper2VecModel(unittest.TestCase):

    def test_paper2vec_init(self):
        p2v = paper2vec.Paper2Vec(papers_file = datapath_content, citation_graph_file = datapath_cites)

    def test_load_data(self):
        p2v = paper2vec.Paper2Vec()
        p2v.load_data(papers_file = datapath_content, citation_graph_file = datapath_cites)

    #def test_train_from_files(self):
    #    d2v_dict = {'alpha': 0.025, 'window': 10, 'min_count': 10, 'min_alpha': 0.025, 'size': 100}
    #    w2v_dict = {'size': 100, 'window': 5}
    #    p2v = paper2vec.Paper2Vec(papers_file = datapath_content, citation_graph_file = datapath_cites,
    #    d2v_dict = d2v_dict, w2v_dict = w2v_dict)
    #    p2v.train()

    def test_train(self):
        p2v = paper2vec.Paper2Vec(papers = contents_test, citation_graph = citation_test,
        d2v_dict = d2v_params, w2v_dict = w2v_params)
        p2v.train()

    def test_predict(self):
        pass

if __name__ == "__main__":
    unittest.main()