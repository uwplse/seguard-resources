import networkx as nx
import subprocess
from main import parSet
from main import strToVec
import numpy as np
def fabricate():
    G = nx.Graph()
    G.add_nodes_from(range(21))
    for i in range(20):
        G.add_edge(i, i + 1)
    nx.write_edgelist(G, "distance_test/output.emb", data=False)


def to_vector(data):
    command = "./node2vec -i:" + "distance_test/output.emb" + \
        " -o:" + "distance_test/output_2.emb"
    command += ' -d:' + str(data.dim)
    command += ' -l:' + str(data.walk)
    command += ' -r:' + str(data.num_walk)
    command += ' -p:' + str(data.p)
    command += ' -q:' + str(data.q)
    subprocess.call(command, shell=True)

def read_p():
    ans = {}
    fres = open("distance_test/output_2.emb", 'r')

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

def main():
    # arbitrary parameter set
    par = parSet(
        dim=500,
        walk=15,
        num_walk=5,
        q=0.25,
        p=0.5
    )
    fabricate()
    to_vector(par)
    d = read_p()


    candidates = [5, 10, 15, 20]
    res = []
    for cand in candidates:
        print(d['1'])
        dis = np.linalg.norm(d['1'], d[str(cand)])
        res.append(dis)
    with open('distance_test/result.txt', 'w') as filehandle:
        for i in range(len(candidates)):
            filehandle.write('%s: %s\n' % (candidates[i], res[i]))

main()
