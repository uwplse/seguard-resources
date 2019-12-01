from experiment import dataset_test_binary
import numpy as np
ran = list(range(1000,10000,1000))


dataset_test_binary(src='synthetic_data',ran=ran)

# from augmented_experiment import fabricate_data

# fabricate_data()