import numpy as np

def subsets(label_set, subset_size):

    """Returns indexes of the initial set where the indexes of the relevant values of the initial set are smaller than the variable subset_size.
    For example, if only the first n notes want to be considered, we set subset_size to be n. 
    Note that the label set must be used as it has the ideal format."""
    smaller_set = []
    for i in range(len(label_set)):
        locations = (np.where(label_set[i]>1)[0])
        if all(j < subset_size for j in locations): 
            smaller_set.append(i)
    return smaller_set