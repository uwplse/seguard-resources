import networkx as nx
import subprocess
from main import parSet
from main import strToVec
import numpy as np

from distance_test import to_vector
from distance_test import read_p

def fabricate_adjacent():
    G = nx.Graph()
    G.add_nodes_from(range(5))
    G.add_edge(0,1)
    G.add_edge(0,2)
    G.add_edge(0,3)
    G.add_edge(0,4)
    nx.write_edgelist(G, "distance_test/output_3.edgelist", data=False)

def main():
    # arbitrary parameter set
    par = parSet(
        dim=25,
        walk=15,
        num_walk=30,
        p=0.2,
        q=0.5
    )
    fabricate_adjacent()
    to_vector(par)
    d = read_p()

    candidates = [1, 2, 3, 4]
    res = []
    for cand in candidates:
        dis = np.linalg.norm(np.array(d['0']) - np.array(d[str(cand)]))
        res.append(dis)
    with open('distance_test/result_2.txt', 'w') as filehandle:
        for i in range(len(candidates)):
            filehandle.write('%s: %s\n' % (candidates[i], res[i]))
