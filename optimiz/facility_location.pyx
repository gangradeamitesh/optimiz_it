import numpy as np
cimport numpy as cnp
from sklearn.metrics import pairwise_distances


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
        for i in range(self.n):
            if i in self.selected_indices:
                continue
            gains[i] = np.maximum(self.X_simi[i] , self.current_values).sum()
        return gains

    def compute_simi_matrix(self) -> np.ndarray:
        X_pairwise = pairwise_distances(self.X, metric="euclidean", squared=True)
        return X_pairwise.max() - X_pairwise
    def select(self, int subset_size) -> np.ndarray: 
        cdef cnp.ndarray[cnp.float64_t , ndim=1] gains
        for _ in range(subset_size):
            gains = self.calculate_gain()
            best_idx = np.argmax(gains)
            self.selected_indices.add(best_idx)
            self.current_values = np.maximum(self.current_values , self.X_simi[best_idx])
        return self.selected_indices