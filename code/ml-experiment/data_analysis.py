import pandas as pd
import os
from utility import parSet
import numpy as np
from experiment import process_data
import time
from main import main
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import pickle

# def test(par, trueval):



data = pd.read_csv('result.csv', sep = '\t')
# print(data.head(3))
good_parameters = data[(data['score'] > 0.74) & (data['std'] < 0.2)]

# Create Fabricated dataset (100, 100)
# 
# Binary Classification: Benign and Malware 
# RandomForest 

# read through the source data, Note that the malwares are labelled as 0
# and the benigns are labelled as 1
pwd = os.getcwd()
# default source is in data/graphs
src = os.path.abspath(os.path.dirname(os.path.dirname(pwd))+os.path.sep+".") + os.path.sep + 'data/graphs'
items = {}
for root, dirs, files in os.walk(src):
    ans = root.split("/")[-1]
    for file in files:
        if(ans != 'benign'):
            items.update({file: 0})
        else:
            items.update({file: 1})


# with different parameter set, transform them into vectors
# For every parameter set, tweek the p and q to augment data
# this is the allowed change for the two parameters, use a for loop to loop through 
ran = [-0.02, -0.01, 0, 0.01, 0.02]

df = pd.DataFrame({
    "dim":[],
    "walk":[],
    "num_walk":[],
    "p":[],
    "q":[],
    "score":[],
    "std":[]
})

for index, row in good_parameters.iterrows():
    # for every fabricated result, read the vectors into array,
    # and the the true value 
    X_data = np.array([]).reshape(0, 3021)

    X_true = []
    for r_1 in ran:
        for r_2 in ran:
            par = parSet(
                dim = row['dim'], 
                walk = row['walk'], 
                num_walk = row['num_walk'], 
                q = row['q'] + r_1, 
                p = row['p'] + r_2
            )
            main(par)
            with open('final_result.pickle', 'rb') as handle:
                vecs = pickle.load(handle)
            X_data_t, X_true_t = process_data(vecs, items)
            X_data = np.vstack((X_data, X_data_t))
            X_true.extend(X_true_t.tolist())
            # print(X_data.shape)
            # print(len(X_true))

            # time.sleep(10)
    clf = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=0)
    scores = cross_val_score(clf, X_data, np.array(X_true), cv=10)
    
    temp1 = scores.mean()
    temp2 = scores.std()
    df2 = pd.DataFrame({
        "dim":row['dim'],
        "walk":row['walk'],
        "num_walk":row['num_walk'],
        "p":row['q'],
        "q":row['p'],
        "score":[temp1],
        "std":[temp2]
    } 
    )
    df = df.append(df2, ignore_index=True)

export_csv = df.to_csv('experiment_2.csv', sep='\t')

    

# test using random forest and cross_validation 