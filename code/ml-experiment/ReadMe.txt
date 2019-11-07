This repo is the python implementation of one-hot-encoding + distance vector for 
Android malware detection. 

To use node2vec to turn dot files into vectors, 

================================================================
main.py scans through all the graphs in data/graph, and turns them into
edgelists in the "/graph". It calls node2vec to turn the graphs into vectors "/emb".
main.py returns a dict consists of node_one_hot and edge_one_hot and node distance (normalized)
in a pickle file called final_result.pickle. 

For the purpose of experiment, one can change the parameters passed to the node2vec in the following way:

python3 main.py --dimensions 128 --walk-length 10 --num-walks 10 --p 1 --q 1
