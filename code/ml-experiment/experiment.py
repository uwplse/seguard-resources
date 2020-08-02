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
from collections import Counter
import networkx as nx
import subprocess
from main import strToVec
import seaborn as sns



from sklearn.metrics import accuracy_score
# scans through the graphs in data/graphs, and denote their real values
# filename -> label, then list(label), sort, -> filename -> loc[label]


# merge some labels: warn_trojan, block_trojan, trojan -> trojan
# returns a dict with filename and appropriate labels.
def true_val(src='data/graphs', merge=False):
    merge_map = {'TrojanSMS.Hippo':'sms', 
                'gooddroid':'benign', 
                'ExploitLinuxLotoor':'backdoor',
                'Lotoor':'backdoor',
                'jssmsers':'sms', 
                'block_hostile_downloader':'downloader',
                'warn_phishing':'phishing',
                'warn_trojan':'trojan',
                'block_phishing':'phishing',
                'warn_sms_fraud':'sms', # double check
                'warn_commercial_spyware':'spyware',
                'warn_spyware':'spyware',
                'warn_backdoor':'backdoor',
                'block_trojan':'trojan',
                'block_backdoor':'backdoor',
                'warn_hostile_downloader':'downloader',
                }
    omit = {'warn_toll_fraud', 'warn_click_fraud','warn_privilege_escalation'}
    res = {}
    pwd = os.getcwd()
    # default source is in data/graphs
    src = os.path.abspath(os.path.dirname(
        os.path.dirname(pwd))+os.path.sep+".") + os.path.sep + src
    for root, dirs, files in os.walk(src):
        ans = root.split("/")[-1]
        for file in files:
            res.update({file : ans})
    
    if merge:
        merged = {}
        for key,value in res.items():
            if value in merge_map.keys():
                merged.update({key:merge_map[value]})
            elif value not in omit:
                merged.update({key:value})
        return merged
    else:
        return res

#  both vecs and targets are dicts {vec: label}
#  put vecs into matrix form, and target into a list of true values
def process_data(vec1, targets, vec2=None):
    label_list = list(set(targets.values()))
    label_list.sort()

    data_1 = []
    if vec2!= None:
        data_2 = []
    target = []
    for name in list(targets.keys()):
        data_1.append(np.array(vec1[name]))
        if vec2!=None:
            data_2.append(np.array(vec2[name]))
        target.append(label_list.index(targets[name]))
    if vec2!=None:
        return np.array(data_1), np.array(data_2), np.array(target)
    else:
        return np.array(data_1), np.array(target)


def evaluate(name,trueval):
    with open(name, 'rb') as handle:
        vecs = pickle.load(handle)
    X_data, X_true = process_data(vecs, trueval)
    clf = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=0)
    scores = cross_val_score(clf, X_data, X_true, cv=10)
    
    temp1 = scores.mean()
    temp2 = scores.std()
    return temp1, temp2

def test(params, trueval,src='data/graph'):
    main(params,src=src)
    res_m = []
    res_s = []
    for i in range(50):
        mean, std = evaluate('final_result.pickle', trueval)
        res_m.append(mean)
        res_s.append(std)
    return sum(res_m)/len(res_m), sum(res_s)/len(res_s)

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
        dim=25,
        walk=15,
        num_walk=30,
        p=0.2,
        q=0.5
    )
    t = true_val(src=src)

    main(par, src=src)
    prev_method(src=src)
    new_mean, new_std = evaluate('final_result.pickle', t)
    prev_mean, prev_std = evaluate('final_result_prev.pickle', t)
    
    with open('compare_result.txt','w') as handle:
        string = "new_mean: {}, new_std: {}, prev_mean: {}, prev_std: {}".format(new_mean, new_std, prev_mean, prev_std) 
        handle.write(string)

def draw(size, lib_new, lib_prev, trueval):
    # print("test")
    # print(len(lib_new))
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

def tru_bin(src='data/graphs',merge=False):
    d = true_val(src=src, merge=merge)
    res = {name:'malware' if label !='benign' else 'benign' for name,label in d.items()}
    return res

# test the accuracy based on the size of the dataset 
def dataset_test_binary(src='metadata', fn=tru_bin, cv=10,name='Binary',ran=[50, 100, 150, 200, 250, 314]):
    par = parSet(
        dim=25,
        walk=15,
        num_walk=30,
        p=0.2,
        q=0.5
    )
    main(par,src=src)
    prev_method(src=src)
    t = fn(src=src,merge=False)
    lib_prev = reading_lib('final_result_prev.pickle')
    lib_new = reading_lib('final_result.pickle')

    diff = list(set(lib_prev.keys()) - set(t.keys()))
    for d in diff:
        del lib_prev[d]
        del lib_new[d]
    new = []
    prev = []
    for ran_1 in ran:
        temp1 = []
        temp2 = []
        for ran_2 in range(100):
            selected_vecs, selected_prev, selected_tru = draw(size=ran_1, lib_new=lib_new, lib_prev=lib_prev, trueval=t)
            selected_vecs, selected_prev, selected_tru_w = process_data(selected_vecs, selected_tru, selected_prev)
            clf = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=0)
            scores_new = cross_val_score(clf, selected_vecs, selected_tru_w, cv=cv)

            clf_2 = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=0)
            scores_prev = cross_val_score(clf_2, selected_prev, selected_tru_w, cv=cv)
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

def dataset_test_multivariate(src='metadata'):
    dataset_test_binary(src=src, fn = true_val,cv=2, name='Multivariate')


def grid_search():
    t = true_val(src='metadata',merge=True)

    dimSet = [5, 10, 25, 50, 70, 100, 128, 200, 250]
    mean = []
    std = []
    for dim in dimSet:
        par = parSet(
            dim=25,
            walk=15,
            num_walk=30,
            p=0.2,
            q=0.5
        )
        mean_t, std_t = test(par, t,src="metadata")
        mean.append(mean_t)
        std.append(std_t)
    plt.figure(1)
    plt.plot(dimSet, mean)
    plt.xlabel('dimension')
    plt.ylabel('accuracy')
    plt.title('dimension vs accuracy')
    plt.show()

    plt.figure(2)
    plt.plot(dimSet, std)
    plt.xlabel('dimension')
    plt.ylabel('standard deviation of accuracy')
    plt.title('dimension vs standard deviation of accuracy')
    plt.show()

    print(mean)
    print(std)



def count():
    t = true_val(src='metadata',merge=True)
    return dict(Counter(t.values()))


def fabricate_distance():
    G = nx.Graph()
    G.add_nodes_from(range(21))
    for i in range(20):
        G.add_edge(i, i + 1)
    nx.write_edgelist(G, "distance_test/distance.edgelist", data=False)


def to_vector(data, edgelist="distance_test/output_2.edgelist",output="distance_test/final.emb"):
    command = "./node2vec -i:" + edgelist + " -o:" + output
    command += ' -d:' + str(data.dim)
    command += ' -l:' + str(data.walk)
    command += ' -r:' + str(data.num_walk)
    command += ' -p:' + str(data.p)
    command += ' -q:' + str(data.q)
    subprocess.call(command, shell=True)

def read_p(read_path="distance_test/final.emb"):
    ans = {}
    fres = open(read_path, 'r')

    # needs some unit testing on if node2vec does the right thing
    try:
       a = fres.readlines()
       for i in range(len(a)):
           if i != 0:
               temp = a[i].split(' ', 1)
               ans.update({temp[0]: strToVec(temp[1])})
    finally:
        fres.close()
    return ans

def distance_test():
    par = parSet(
        dim=250,
        walk=15,
        num_walk=100,
        p=0.5,
        q=0.8
    )
    candidates = [1, 5, 10, 15, 20]
    distances = np.zeros((50, 5))
    for i in range(50):
        fabricate_distance()
        to_vector(par,edgelist="distance_test/distance.edgelist", output="distance_test/distance.emb")
        d = read_p(read_path="distance_test/distance.emb")

        for index in range(len(candidates)):
            dis = np.linalg.norm(np.array(d['0']) - np.array(d[str(candidates[index])]))
            distances[i][index] = dis

    sns.set()
    pl = sns.heatmap(np.moveaxis(distances, 0, -1),yticklabels=candidates)
    plt.title("Distance between loosely conneceted nodes under same setting")
    plt.ylabel('Distance from node 0')
    plt.xlabel('trials')
    fig = pl.get_figure()
    fig.savefig("distance_test/distance_graph.png")
    # with open('distance_test/distance_test.txt', 'w') as filehandle:
    #     for i in range(len(candidates)):
    #         filehandle.write('%s: %s\n' % (candidates[i], res[i]))


def fabricate_adjacent(edgelist="distance_test/adjacent.edgelist"):
    G = nx.Graph()
    G.add_nodes_from(range(5))
    G.add_edge(0,1)
    G.add_edge(0,2)
    G.add_edge(0,3)
    G.add_edge(0,4)
    nx.write_edgelist(G, edgelist, data=False)

def adjacent_test():
    # arbitrary parameter set
    par = parSet(
        dim=25,
        walk=15,
        num_walk=30,
        p=0.2,
        q=0.5
    )

    candidates = [1, 2, 3, 4]

    distances = np.zeros((50,4))
    for i in range(50):
        fabricate_adjacent()
        to_vector(par, edgelist="distance_test/adjacent.edgelist",output="distance_test/adjacent.emb")
        d = read_p("distance_test/adjacent.emb")
        for index in range(len(candidates)):
            distances[i][index] = np.linalg.norm(np.array(d['0']) - np.array(d[str(candidates[index])]))
    sns.set()
    pl = sns.heatmap(np.moveaxis(distances, 0, -1),yticklabels=candidates)
    plt.title("Distance between adjacent nodes under same setting")
    plt.ylabel('Distance from node 0')
    plt.xlabel('trials')
    fig = pl.get_figure()
    fig.savefig("distance_test/adjacent_graph.png") 
    
def adj_distribution():
    par = parSet(
        dim=25,
        walk=15,
        num_walk=30,
        p=0.2,
        q=0.5
    )
    res = []

    for i in range(10000):
        fabricate_adjacent("distance_test/adj_dist.edgelist")
        to_vector(par, edgelist="distance_test/adj_dist.edgelist",output="distance_test/adj_dist.emb")
        d = read_p("distance_test/adj_dist.emb")
        dis = np.linalg.norm(np.array(d['0']) - np.array(d['1']))
        res.append(dis)
    print(len(res))

    n, bins, patches = plt.hist(res, 20,facecolor='blue', alpha=0.5)
    plt.xlabel('distance')
    plt.ylabel('count')
    plt.title("adjacent distance distribution")
    plt.show()

    
