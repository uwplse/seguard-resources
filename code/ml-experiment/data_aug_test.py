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
from pathlib import Path
from main import node2vec_mapping
from seguard.graph import Graph

from main import parSet
from main import lib_gen
from main import cGraph

# manually selected range
ran = [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15]

# manually picked a file as a study object
FILE = '3abfa08b4e1de7195c8e9fe52796a37f9a275cb47f6d0fc904eed172061cd56a.apk.top.dot'


def compare(params,vary_q=True):
    res = []
    for ran_1 in ran:
        if vary_q: 
            q = params.q + ran_1
            p = params.p
        else:
            q = params.q
            p = params.p + ran_1
        par = parSet(
            dim=params.dim,
            walk=params.walk,
            num_walk=params.num_walk,
            q=q,
            p=p
        )

        # the number of node distances is 2485 in this dataset
        #
        d, node_lib, edge_lib = lib_gen(par)
        g = d[FILE]
        vec3 = g.distance(node_lib)
        vec = [x for x in vec3 if x != 2.0]
        res.append(vec)
        # 71 nodes
    sns.set()
    ylabel = np.array( ran) + params.q
    pl = sns.heatmap(np.array(res), yticklabels=ylabel,xticklabels=False)
    # pl.set(xlabel=par.__str__())
    fig = pl.get_figure()
    plt.xlabel('dimension')
    plt.title('Different featurization on the same graph')
    if vary_q:
        plt.ylabel('q')
        fig.savefig( 'q: ' + par.__str__() + '.png')
    else:
        plt.ylabel('p')
        fig.savefig( 'p: ' + par.__str__() + '.png')
    fig.clf()


def compare_graph(vary_q=True):
    par = parSet(
        dim=25,
        walk=15,
        num_walk=30,
        p=0.2,
        q=0.5
    )
    compare(par,vary_q=vary_q)

def compare_node_embedding(vary_q=True):
    par = parSet(
        dim=25,
        walk=15,
        num_walk=30,
        p=0.2,
        q=0.5
    )
    root = Path(os.getcwd()).parent.parent
    root = str(root) + os.sep + 'data/graphs/benign/3abfa08b4e1de7195c8e9fe52796a37f9a275cb47f6d0fc904eed172061cd56a.apk.top.dot'
    G = Graph(dot_file=root, config=default_config)
    target = list(G.nodes)[np.random.randint(len(list(G.nodes)))]
    res = []
    for ran_1 in ran:

        if vary_q: 
            p = par.p
            q = par.q + ran_1
        else:
            p =par.p + ran_1
            q = par.q

        par_1 = parSet(dim = par.dim, 
                    walk = par.walk,
                    num_walk = par.num_walk,
                    p=p, 
                    q=q)
        mapping = node2vec_mapping(FILE, G, par_1)
        res.append(mapping[target])

    sns.set()
    if vary_q:
        y_label=np.array(par.q + np.array(ran))
    else:
        y_label= [round(r + par.p,2) for r in ran]
    pl = sns.heatmap(np.array(res),yticklabels=y_label)
    pl.set(xlabel=par.__str__())
    plt.title('Different node embedding on the same node')
    if vary_q:
        plt.ylabel('q')
    else:
        plt.ylabel('p')

    plt.xlabel('dimension')
    fig = pl.get_figure()
    if vary_q: 
        fig.savefig( 'q: ' + par.__str__() + '.png')
    else:
        fig.savefig( 'p: ' + par.__str__() + '.png')

    fig.clf()
