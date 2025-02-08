
from .base_function import BaseFunction
import numpy as np
from scipy.linalg import det , logm
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import pairwise_distances


class LogDeterminant(BaseFunction):

    def __init__(self,X=None, selected_indices=None, gains=None, optimizer_type="naive_greedy" , current_values=None) -> None:
        
        self.X = X
        self.simi_matrix = self.compute_similarity_matrix()
        print(np.array(self.simi_matrix))
        self.n = self.simi_matrix.shape[0]
        self.d = self.simi_matrix.shape[1]
        self.selected_indices = selected_indices if selected_indices is not None else []
        self.optimizer_type = optimizer_type
        self.current_values = 0
        super().__init__(selected_indices, gains, optimizer_type)
    
    def fit(self , subset_size = None):

        return super().fit(subset_size)

    def gain(self , subset):
        K_S =self.simi_matrix[np.ix_(subset , subset)]
        print(K_S)
        print("Printing log det for subset : ", subset)
        print(np.log(det(K_S)))
        return np.log(det(K_S)) if det(K_S)>0 else -np.inf

    def calculate_gain(self):
        gains = np.zeros(self.n , dtype='float64')
        for i in range(self.n):
            #print("Inside the for loop")
            if i in self.selected_indices:
                continue
            gains[i] = self.gain(self.selected_indices+[i]) - self.current_values
            #print(gains[i])
        return gains

    def compute_similarity_matrix(self):
        """Taking the pairwise maxtrix with euclidian distance"""
        X_pairwise_distances = pairwise_distances(self.X , metric="euclidean")
        pairwise = X_pairwise_distances.max() - X_pairwise_distances
        return pairwise
        