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
    temp = list(targets.keys())
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

def compare(src='data/graphs'):
    # an arbitrary good parameter set
    par = parSet(
        dim=250,
        walk=15,
        num_walk=8,
        q=0.25,
        p=4.0
    )
    t = true_val(src=src)

    main(par, src=src)
    prev_method(src=src)
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


def draw(size, lib_new, lib_prev, trueval):
    index = np.random.randint(0, len(lib_new), size=size)
    names = list(lib_new.keys())
    names = [names[i] for i in index.tolist()]
    lib_res = {name : lib_new[name] for name in names}
    tru_res = {name : trueval[name] for name in names}
    lib_prev_res = {name: lib_prev[name] for name in names}
    return lib_res, lib_prev_res, tru_res

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
        # print("ans:-----------")
        ans = root.split("/")[-1]
        # print(ans)
        # print("files:----------")
        print(files)
        for file in files:
            res.update({file : 1 if ans == 'benign' else 0})
    return res

# test the accuracy based on the size of the dataset 
def dataset_test_binary(src='data/graphs', fn=tru_bin, cv=10,name='Binary'):
    ran = [50, 100, 150, 200, 250, 314]
    par = parSet(
        dim=250,
        walk=15,
        num_walk=8,
        q=0.25,
        p=3.0
    )
    main(par,src='metadata')
    prev_method(src='metadata')
    lib_prev = reading_lib('final_result_prev.pickle')
    lib_new = reading_lib('final_result.pickle')
    t = fn(src='metadata')
    s = t.keys()
    new = []
    prev = []
    for ran_1 in ran:
        temp1 = []
        temp2 = []
        for ran_2 in range(10):
            selected_vecs, selected_prev, selected_tru = draw(size=ran_1, lib_new=lib_new, lib_prev=lib_prev, trueval=t)
            selected_vecs, selected_tru_w = process_data(selected_vecs, selected_tru)
            selected_prev, selected_tru = process_data(selected_prev, selected_tru)
            clf = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=0)
            scores_new = cross_val_score(clf, selected_vecs, selected_tru_w, cv=cv)

            clf_2 = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=0)
            scores_prev = cross_val_score(clf, selected_prev, selected_tru_w, cv=cv)
            temp1.append(scores_new.mean())
            temp2.append(scores_prev.mean())
        
        new.append(sum(temp1)/len(temp1))
        prev.append(sum(temp2)/len(temp2))

    # print(mean)
    plt.plot(ran, new, '-g', label='new method')
    plt.plot(ran, prev, '-b', label='previous method')
    plt.legend()
    plt.xlabel("size of dataset")
    plt.ylabel(str(cv) + "-fold cross validation accuracy")
    plt.title( name + " Classification")
    plt.show()

def dataset_test_multivariate():
    dataset_test_binary(src='metadata', fn = true_val,cv=2, name='Multivariate')
