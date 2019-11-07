# tests if the current data augmentation solution makes a
# difference in the node distance part
#

# tests 5 parameter sets, for each set, oscilate in a 2 * 2 manner,
# and outputs 4 heat maps.

import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seguard.common import default_config

from main import node2vec_mapping
from seguard.graph import Graph

from main import parSet
from main import lib_gen
from main import cGraph

# manually selected range
ran = [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]

# manually picked a file as a study object
FILE = '3abfa08b4e1de7195c8e9fe52796a37f9a275cb47f6d0fc904eed172061cd56a.apk.top.dot'


def compare(params):
    res = []
    for ran_1 in ran:
        par = parSet(
            dim=params.dim,
            walk=params.walk,
            num_walk=params.num_walk,
            q=params.q,
            p=params.p + ran_1
        )

        # the number of node distances is 2485 in this dataset
        #
        d, node_lib, edge_lib = lib_gen(par)
        g = d[FILE]
        vec3 = g.distance(node_lib)
        vec = [x for x in vec3 if x != 2.0]
        res.append(vec)
        # 71 nodes
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    print(np.array(res).shape)
    pl = sns.heatmap(np.array(res), yticklabels=np.array(params.q + ran))
    pl.set(xlabel=par.__str__())
    fig = pl.get_figure()
    fig.savefig( 'q: ' + par.__str__() + '.png')
    fig.clf()


def compare_graph():
    data = pd.read_csv('result.csv', sep='\t')
    good_parameters = data[(data['score'] > 0.74) & (data['std'] < 0.2)]
    for index, row in good_parameters.head(n=5).iterrows():
        par = parSet(
            dim=row['dim'],
            walk=row['walk'],
            num_walk=row['num_walk'],
            q=row['q'],
            p=row['p']
        )
        compare(par)

def compare_node_embedding():
    par = parSet(dim = 25, 
                    walk = 15,
                    num_walk = 8,
                    p=0.25, 
                    q=4.0)
    
    G = Graph(dot_file=root + os.sep + FILE, config=default_config)
    target = list(G.nodes)[np.random.randint(len(list(G.nodes)))]
    res = []
    for ran_1 in range():
        par = parSet(dim = 25, 
                    walk = 15,
                    num_walk = 8,
                    p=0.25 + ran_1, 
                    q=4.0)
        mapping = node2vec_mapping(FILE, G, par)
        res.append(mapping[target])
    
    sns.set()
    pl = sns.heatmap(np.array(res), yticklabels=np.array(par.p + ran))
    pl.set(xlabel=par.__str__())
    fig = pl.get_figure()
    fig.savefig( 'p: ' + par.__str__() + '.png')
    fig.clf()
