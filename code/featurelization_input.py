from seguard.common import default_config
import re
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pygraphviz

from tools.python.seguard.graph import Graph
fh = open("test_edge.edgelist",'w')

test_graph =  Graph(dot_file="31a6a74c272af95f5223602e12ef1364527f584675445530eee060e635d2cafb.apk.top.dot", config=default_config)
l = list(test_graph.nodes)
with open('mapping.txt', 'w') as f:
    for item in l:
        f.write("%s\n" % item)
converted = nx.relabel.convert_node_labels_to_integers(test_graph.g, first_label = 1)
# Mapping: the node with position p in list(G.nodes()) is mapped to p+1 in the new graph

nx.write_edgelist(converted, "test_edge.edgelist")
fh.close()
fh = open("test_edge.edgelist","r")
fres = open("result.edgelist", 'w')
for line in fh:
    y = re.sub("[\{].*[\}]", "", line)
    fres.write(y)
fh.close()
fres.close()
