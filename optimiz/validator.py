import numpy as np

def validate_simi_matrix(simi_matrix):
    if simi_matrix==None:
        raise "Similarity Matrix is None"

    if simi_matrix.shape[0]!=simi_matrix.shape[1]:
        raise "Similarity Matrix is not a Square Matrix."
    
    if not isinstance(simi_matrix , np.ndarray):
        raise "Similarity Matrix should be instance of a Numpy array."
    
