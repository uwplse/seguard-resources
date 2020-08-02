
from seguard.graph import Graph
import networkx
import numpy as np
import os
from seguard.common import default_config

def adjust(base):
    nodes = list(base.nodes)
    index = np.random.randint(len(nodes))
    base.remove_node(nodes[index])
    return base

# networkx.drawing.nx_agraph.write_dot
def fabricate_data(src='/metadata'):
    ID = 0
    # ID2F = {}
    pwd = os.getcwd()
    
    for root, dirs, files in os.walk(pwd+src):
        label = root.split("/")[-1]
        for file in files:
            G = Graph(dot_file=root + os.sep + file, config=default_config)
            raw_nx_graph = G.g
            for i in range(30):
                g_adjusted = adjust(raw_nx_graph.copy())
                path = pwd + os.path.sep + 'synthetic_data/'+ label
                if not os.path.exists(path):
                    os.makedirs(path)

                networkx.drawing.nx_agraph.write_dot(g_adjusted, path  + os.path.sep + str(ID) + '.dot')
                ID += 1


            
            