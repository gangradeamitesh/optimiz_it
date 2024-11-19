import numpy as np
cimport numpy as cnp
from sklearn.metrics import pairwise_distances
from cython.parallel import prange


cdef parallel_get_gain(int n ,double[: , :] simi_x ,double[:] current_values):
    #cdef double[:] gains = np.zeros(n , dtype='float64')
    cdef int i , j
    cdef double max_value , gain_sum
    cdef double[:] partial_gain = np.zeros(n , dtype='float64')
    cdef int current_values_len
    for i in prange(n , nogil=True):
        current_values_len = len(current_values)
        #gain_sum = 0.0
        for j in range(current_values_len):
            max_value = max(simi_x[i, j], current_values[j])
            partial_gain[i] += max_value
        #partial_gain[i] = gain_sum
    return partial_gain

# cdef get_sum(double[:] simi_x , double[:] current_values):
#     cdef double gain_i
#     gain_i = np.maximum(np.asarray(simi_x) , np.asarray(current_values)).sum()
#     return gain_i



class FacilityLocation:
    def __init__(self , cnp.ndarray[cnp.float64_t , ndim=2] X , cnp.ndarray[cnp.int64_t , ndim=1] y=None) -> None:

        self.X = X
        self.X_simi = self.compute_simi_matrix()
        self.n = self.X_simi.shape[0]
        self.d = self.X_simi.shape[1]
        self.current_values = np.zeros(self.d, dtype='float64')
        #cdef cnp.ndarray[cnp.int64_t , ndim=1] idx = np.zeros(self.n, dtype='int64')
        self.selected_indices = set()

    def calculate_gain(self) -> np.ndarray:
        cdef cnp.ndarray gains = np.zeros(self.n , dtype = 'float64')
        n = self.n
        for i in range(n):
            if i in self.selected_indices:
                continue
            gains[i] = np.maximum(self.X_simi[i] , self.current_values).sum()
        return gains
        
        #return parallel_get_gain(self.n , self.X_simi , self.current_values)

    def compute_simi_matrix(self) -> np.ndarray:
        X_pairwise = pairwise_distances(self.X, metric="euclidean", squared=True)
        return X_pairwise.max() - X_pairwise
    def select(self, int subset_size) -> np.ndarray: 
        cdef cnp.ndarray[cnp.float64_t , ndim=1] gains
        for _ in range(subset_size):
            #gains = np.asarray(self.calculate_gain())
            gains = np.asarray(self.calculate_gain())
            best_idx = np.argmax(gains)
            self.selected_indices.add(best_idx)
            self.current_values = np.maximum(self.current_values , self.X_simi[best_idx])
        return self.selected_indices