import os
from seguard.common import default_config
import re
import argparse
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pygraphviz
import copy 
from sklearn.feature_extraction import DictVectorizer
import pickle
import subprocess
import time
from tqdm import tqdm

from seguard.graph import Graph

'''
1. for each graph, create a map from the node name to its node2vec vector.
2. for each graph, add nodes to dict1 [name ->  Method name, value -> 1], 
edges to dict2 [nodeA+nodeB -> Name, value -> 1]), union to create the library
3. For each graph G', using the library, generate node one-hot vector. (DictVectorizer). 
4. edge one-hot vector.(DictVectorizer)
5. node distance vector -> cartesian distance (N choose 2)
6. concatenate and output
'''



'''
returns a map from the node name to corresponding node2vec vector
this function should be parametrized: self defined path to node2vec
The node2vec command/flags should be arguments instead of default
Note: this is using c++ version instead of python
'''
def node2vec_mapping(name, arg, data):
    l = copy.deepcopy(list(arg.nodes))
    converted = nx.relabel.convert_node_labels_to_integers(arg.g, first_label = 1)

    edge_name = "graph/" + name + ".edgelist"
    result_name = "emb/" + name + ".emb"
    fh = open(edge_name,'w')

    nx.write_edgelist(converted, edge_name, data=False)

    command = "./node2vec -i:" + edge_name + " -o:" + result_name
    command += ' -d:' + str(data.dim)
    command += ' -l:' + str(data.walk)
    command += ' -r:' + str(data.num_walk)
    command += ' -p:' + str(data.p)
    command += ' -q:' + str(data.q)
    subprocess.call(command, shell=True)

    # suppose the resulting emb has the same name as the 
    ans = {}
    fres = open(result_name, 'r')

    # needs some unit testing on if node2vec does the right thing
    try:
       a = fres.readlines()
       for i in range(len(a)):
           if i != 0:
               temp = a[i].split(' ', 1)
               ans.update({l[int(temp[0]) - 1]: strToVec(temp[1])})
    finally:
        fres.close()
    
    return ans

def strToVec(arg):
    ans = arg.split()
    for i in range(len(ans)):
        ans[i] = float(ans[i])
    return ans

def lib_gen(args, src='data/graphs'):
    pwd = os.getcwd()
    # default source is in data/graphs
    if(src == 'data/graphs'):
        src = os.path.abspath(os.path.dirname(os.path.dirname(pwd))+
            os.path.sep+".") + os.path.sep + src
    print("src: " + str(src))
    d = {}
    node_lib = set()
    edge_lib = set()
    for root, dirs, files in os.walk(src):
        for file in files:
            # print("file name: " + str())
            filename, file_extension = os.path.splitext(root + os.sep + file)
            if file_extension == '.dot':
                G = Graph(dot_file=root + os.sep + file, config=default_config)
                # methodName -> vector
                temp = node2vec_mapping(file, G, args)
                # methodName -> 1
                temp_nodes = {x : 1 for x in list(G.nodes)}
                # (methodName, methodName) -> 1
                temp_edges = {x : 1 for x in list(G.edges)}

                node_lib = node_lib.union(G.nodes)
                edge_lib = edge_lib.union(G.edges)
                d.update({file: cGraph(graph = G, vec = temp, nodes = temp_nodes, edges = temp_edges)}) 
    
    node_lib = {x : 1 for x in node_lib}
    edge_lib = {x : 1 for x in edge_lib}

    return d, node_lib, edge_lib
    

# scanns the src and turn them into vectors into final_result.pickle
def main(args, src='data/graphs'):
    pwd = os.getcwd()
    # default source is in data/graphs
    src = os.path.abspath(os.path.dirname(os.path.dirname(pwd))+
            os.path.sep+".") + os.path.sep + src
    d, node_lib, edge_lib = lib_gen(args, src=src)

    result = {}
    for root, dirs, files in os.walk(src):
        for file in files:
            filename, file_extension = os.path.splitext(root + os.sep + file)
            if file_extension == '.dot':
                g = d[file]
                vec1 = g.node_one_hot(node_lib)
                vec2 = g.edge_one_hot(edge_lib)
                vec3 = g.distance(node_lib)
                res =  np.concatenate((vec1, vec2, vec3),axis = None)
                result.update({file: res })

    with open('final_result.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
class cGraph(object):
    def __init__(self, graph=None, vec=None, nodes=None, edges=None):
        self.g = graph
        self.vec = vec
        self.nodes = nodes
        self.edges = edges

    
    def node_one_hot(self, lib):
        v = DictVectorizer(sparse=False)
        lib1 = []
        lib1.append(lib)
        v.fit(lib1)
        X = v.transform(self.nodes)
        return X[0]

    def edge_one_hot(self, lib):
        v = DictVectorizer(sparse=False)
        lib1 = []
        lib1.append(lib)
        v.fit(lib1)
        return v.transform(self.edges)[0]

    def distance(self, lib):
        res = []
        nodes = list(lib.keys())
        nodes.sort()
        for i in range(len(nodes)):
            node1 = nodes[i]
            j = i + 1
            while j < len(nodes):
                node2 = nodes[j]
                try:
                    vec1 = np.array(self.vec[node1])
                    vec2 = np.array(self.vec[node2])

                except KeyError:
                    # a normalized distance should always be positive, assigning a negative value
                    # means either of the nodes does not exist
                    res.append(-10.0)
                    j += 1

                else:
                    dis = np.linalg.norm(vec1 - vec2)
                    res.append(dis)
                    j += 1

        res = np.array(res)
        
        res /= max(res)

        for i in range(len(res)):
            if res[i] < 0:
                res[i] = 2.0
        
        return res

# the class parSet defines the parameters we are fine tuning
class parSet:
    def __init__(self, dim, walk, num_walk, q, p):
        self.dim = dim
        self.walk = walk
        self.num_walk = num_walk
        self.q = q
        self.p = p
    def __str__(self):
        return str(self.dim) + '_' + str(self.walk) + '_' + str(self.num_walk) + '_' + str(self.p) + '_' + str(self.q)

def parse_args():
    parser = argparse.ArgumentParser(description="Featurelization.")
    parser.add_argument('--dim', type=int, default=128, help='Number of dimensions. Default is 128.')
    parser.add_argument('--walk', type=int, default=80, help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walk', type=int, default=10, help='Number of walks per source. Default is 10.')
    parser.add_argument('--p', type=float, default=1, help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=1, help='Inout hyperparameter. Default is 1.')
    return parser.parse_args()

# if __name__ == "__main__":
#     args = parse_args()
#     main(args)
