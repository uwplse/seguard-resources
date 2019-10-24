import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from main import parSet
from main import main
from main import cGraph
import pandas as pd
from seguard.graph import Graph
from seguard.common import default_config
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
# scans through the graphs in data/graphs, and denote their real values
# filename -> label, then list(label), sort, -> filename -> loc[label]

def true_val(src='data/graphs'):
    res = {}
    pwd = os.getcwd()
    # default source is in data/graphs
    src = os.path.abspath(os.path.dirname(
        os.path.dirname(pwd))+os.path.sep+".") + os.path.sep + src
    for root, dirs, files in os.walk(src):
        ans = root.split("/")[-1]
        for file in files:
            res.update({file : ans})
    temp = set(res.values())
    temp = list(temp)
    temp.sort()
    res2 = {i: temp.index(res[i]) for i in list(res.keys())}
    return res2

#  both vecs and targets are dicts 
def process_data(vecs, targets):
    temp = list(vecs.keys())
    temp.sort()

    data = []
    target = np.array([])
    for name in temp:
        temp = np.array(vecs[name])
        data.append(temp)
        target = np.concatenate((target, targets[name]), axis=None)
    
    data = np.array(data)
    return data, target


def evaluate(name,trueval=None):
    with open(name, 'rb') as handle:
        vecs = pickle.load(handle)
    X_data, X_true = process_data(vecs, trueval)
    clf = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=0)
    scores = cross_val_score(clf, X_data, X_true, cv=10)
    
    temp1 = scores.mean()
    temp2 = scores.std()
    return temp1, temp2

def test(params, trueval):
    main(params)
    mean, std = evaluate('final_result.pickle', trueval)
    df2 = pd.DataFrame({
        "dim":[params.dim],
        "walk":[params.walk],
        "num_walk":[params.num_walk],
        "p":[params.p],
        "q":[params.q],
        "score":[mean],
        "std":[std]
    } 
    )

    return df2

# previous method that only uses node-one-hot encoding and edge-one-hot encoding
def prev_method(src='data/graphs'):
    pwd = os.getcwd()
    # default source is in data/graphs
    src = os.path.abspath(os.path.dirname(os.path.dirname(pwd))+
            os.path.sep+".") + os.path.sep + src
    d = {}
    node_lib = set()
    edge_lib = set()
    for root, dirs, files in os.walk(src):
        for file in files:
            filename, file_extension = os.path.splitext(root + os.sep + file)
            if file_extension == '.dot':
                G = Graph(dot_file=root + os.sep + file, config=default_config)
                # methodName -> 1
                temp_nodes = {x : 1 for x in list(G.nodes)}
                # (methodName, methodName) -> 1
                temp_edges = {x : 1 for x in list(G.edges)}

                node_lib = node_lib.union(G.nodes)
                edge_lib = edge_lib.union(G.edges)
                d.update({file: cGraph(graph = G, nodes = temp_nodes, edges = temp_edges)}) 
    
    node_lib = {x : 1 for x in node_lib}
    edge_lib = {x : 1 for x in edge_lib}

    result = {}
    for root, dirs, files in os.walk(src):
        for file in files:
            filename, file_extension = os.path.splitext(root + os.sep + file)
            if file_extension == '.dot':
                g = d[file]
                vec1 = g.node_one_hot(node_lib)
                vec2 = g.edge_one_hot(edge_lib)
                res =  np.concatenate((vec1, vec2),axis = None)
                result.update({file: res })

    with open('final_result_prev.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

def compare():

    # an arbitrary good parameter set
    par = parSet(
        dim = 500, 
        walk = 15, 
        num_walk = 5, 
        q = 0.25,
        p = 0.5
    )
    t = true_val(src='metadata')

    main(par, src='metadata')
    prev_method(src='metadata')
    new_mean, new_std = evaluate('final_result.pickle', t)
    prev_mean, prev_std = evaluate('final_result_prev.pickle', t)
    
    with open('compare_result.txt','w') as handle:
        string = "new_mean: {}, new_std: {}, prev_mean: {}, prev_std: {}".format(new_mean, new_std, prev_mean, prev_std) 
        handle.write(string)


def grid_search():
    t = true_val()

    dimSet = [50, 70, 100, 128, 200, 250, 300, 500]
    walkSet= [5, 10, 15, 20, 30]
    num_walkSet = [5, 10, 15, 20, 30]
    qSet = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    pSet = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]


    df = pd.DataFrame({
        "dim":[],
        "walk":[],
        "num_walk":[],
        "p":[],
        "q":[],
        "score":[],
        "std":[]
    })

    for dim in dimSet:
        for walk in walkSet:
            for num_walk in num_walkSet:
                for q in qSet:
                    for p in pSet:
                        par = parSet(dim = dim, 
                                    walk = walk,
                                    num_walk = num_walk,
                                    q = q, 
                                    p = p)
                        
                        df2 = test(par, t)
                        df = df.append(df2, ignore_index=True)


    export_csv = df.to_csv('result.csv', sep='\t')


def draw(size, lib, trueval):
    index = np.random.randint(0, len(lib), size=size)
    names = lib.keys()
    names = names[index]
    lib_res = {name : lib[name] for name in names}
    tru_res = {name : trueval[name] for name in names}
    return lib_res, tru_res

def reading_lib(file):
    with open(file, 'rb') as handle:
        vecs = pickle.load(handle)
    return vecs

def tru_bin(src='data/graphs'):
    res = {}
    pwd = os.getcwd()
    # default source is in data/graphs
    src = os.path.abspath(os.path.dirname(
        os.path.dirname(pwd))+os.path.sep+".") + os.path.sep + src
    for root, dirs, files in os.walk(src):
        ans = root.split("/")[-1]
        for file in files:
            res.update({file : 0 if ans == 'benign' else 0})
    return res


# test the accuracy based on the size of the dataset 
def dataset_test_binary():
    ran = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    par = parSet(
        dim = 500, 
        walk = 15, 
        num_walk = 5, 
        q = 0.25,
        p = 0.5
    )
    main(par)
    lib = reading_lib('final_result.pickle')
    t = tru_bin(src='metadata')
    s = t.keys()
    mean = []
    std = []
    for ran_1 in ran:
        selected_vecs, selected_tru = draw(size=ran_1, lib=lib, trueval=t)
        clf = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=0)
        scores = cross_val_score(clf, selected_vecs, selected_tru, cv=10)
        
        temp1 = scores.mean()
        temp2 = scores.std()
        mean.append(temp1)
        std.append(temp2)
    
    plt.plot(ran, mean, )

def dataset_test_multivariate():
    ran = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
