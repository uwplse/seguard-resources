
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
