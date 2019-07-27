import pickle
DX = pickle.load(open("DX.p", "rb"))
DY = pickle.load(open("DY.p", "rb"))
DZ = pickle.load(open("DZ.p", "rb"))


# Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=0)
clf.fit(DX, DZ)
print(clf.feature_importances_)
scores = cross_val_score(clf, DX, DZ, cv=10)
print(scores)

# SVC
from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(DX, DY) 
scores = cross_val_score(clf, DX, DY, cv=10)
print(scores)

# K neighbors
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(DX, DZ)
scores = cross_val_score(clf, DX, DZ, cv=10)
print(scores)