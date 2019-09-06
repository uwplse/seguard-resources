This repo is the python implementation of one-hot-encoding + distance vector for 
Android malware detection. 

Note that all the code must be run from the directory seguard_resources/code/ml-experiment

For non mac-os users, generate excutables from https://github.com/snap-stanford/snap/tree/master/examples/node2vec
and replace the node2vec executable in the repo

To use node2vec to turn dot files into vectors, one should use import main from main.py. 
main accepts a parameter set as described in the blog post. One can either define
their own parameter set, which has the following parameter set defined: 
dim, walk, num_walk, q, p. Or one can use the parSet class defined in main.py. 

main also accepts the path to the dot files to be transformed. If not specified, 
main looks for file under seguard_resources/data/graphs. 

main generates a pickle file called final_result.pickle, which contains featurized graph
imformation in a python dictionary. Structured as follows: {<filename>:<file_featurization>}

================================================================

To run experiments conducted in the blog post:

Node2Vec Hyper-Parameter selection:
    from experiment import grid_search. grid_search is a wrapped method that searches dimensions
    and outputs with two graphs: (10-fold cross-validation) accuracy mean and accuracy standard 
    deviation. 

Data Augmentation for Validation:
    from data_aug_test import compare_node_embedding. Note that the file comparing is hardcoded,
    one can change to other files on the top of the scipt. compare_node_embedding accepts a bool
    vary_q. if vary_q, compare_node_embedding generates graph by perturbing q. 

Comparison between previous method and new method on binary and multivariate classification:
    from experiment import dataset_test_binary. Default path to file is seguard_resources/data/graphs.
    The function dataset_test_binary generates a graph demonstrating the effect of change of 
    dataset size to binary classification accuracy. 
    The function dataset_test_multivariate generates a similar graph for multivariate classification. 