import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

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
    # print(temp)
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


trueval = true_val()
with open('final_result.pickle', 'rb') as handle:
    vecs = pickle.load(handle)
X_data, X_true = process_data(vecs, trueval)
# print(X_data.shape)
# print(X_true.shape)
clf = RandomForestClassifier(n_estimators=1000, max_depth=50, random_state=0)
# scores = cross_val_score(clf, X_data, X_true, cv=10)
# print(scores)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

y_pred = cross_val_predict(clf, X_data, X_true, cv = 10)
print(accuracy_score(X_true, y_pred))


# flow -> 
#  选一个参数，然后在一个大概10个大小的list里面iterate， 每一次iterate都会call一次featurelization code
#  然后process data，做以上的following， 然后把参数集和accracy result写在一个pickle文件里，使用dict
