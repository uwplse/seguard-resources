import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from utility import parSet
from main import main
import pandas as pd

from sklearn.metrics import accuracy_score
# scans through the graphs in data/graphs, and denote their real values
# filename -> label, then list(label), sort, -> filename -> loc[label]

def true_val():
    res = {}
    pwd = os.getcwd()
    # default source is in data/graphs
    src = os.path.abspath(os.path.dirname(os.path.dirname(pwd))+os.path.sep+".") + os.path.sep + 'data/graphs'
    for root, dirs, files in os.walk(src):
        ans = root.split("/")[-1]
        for file in files:
            res.update({file : ans})
    temp = set(res.values())
    temp = list(temp)
    temp.sort()
    res2 = {i: temp.index(res[i]) for i in list(res.keys())}
    return res2

#  now, we have feature vectors and true values, use random forest, which appeared to be the best
#  clf in the last experiment
#  need to first transform the data into the desired forms 


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


def test(params, trueval):
    main(params)
    with open('final_result.pickle', 'rb') as handle:
        vecs = pickle.load(handle)
    X_data, X_true = process_data(vecs, trueval)
    clf = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=0)
    scores = cross_val_score(clf, X_data, X_true, cv=10)
    
    temp1 = scores.mean()
    temp2 = scores.std()
    df2 = pd.DataFrame({
        "dim":[params.dim],
        "walk":[params.walk],
        "num_walk":[params.num_walk],
        "p":[params.p],
        "q":[params.q],
        "score":[temp1],
        "std":[temp2]
    } 
    )

    return df2


def scpt():
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
                        par = parSet(dim = dim, walk = walk, num_walk = num_walk, q = q, p = p)
                        
                        df2 = test(par, t)
                        df = df.append(df2, ignore_index=True)


    export_csv = df.to_csv('result.csv', sep='\t')


