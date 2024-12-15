
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
        self.selected_indices = selected_indices if selected_indices is not None else set()
        self.optimizer_type = optimizer_type
        self.current_values = current_values if current_values is not None else np.zeros(self.d, dtype='float64')
        super().__init__(selected_indices, gains, optimizer_type)
    
    def fit(self , subset_size = None):

        return super().fit(subset_size)

    def log_det(self , subset):
        print(subset)
        K_S =self.simi_matrix[np.ix_(subset , subset)]
        print(K_S)
        return np.log(det(K_S)) if det(K_S)>0 else -np.inf

    def calculate_gain(self):
        gains = np.zeros(self.n , dtype='float64')
        for i in range(self.n):
            #print("Inside the for loop")
            # if i not in self.selected_indices:
            #     #continue
            gains[i] = self.log_det(list(self.selected_indices)+[i]) - self.log_det(list(self.selected_indices))
            print(gains[i])
        return gains

    # def compute_simi(self):
    #     """Implementing only for rbf but other kernels can be used too"""
    #     # if kernel_type=='rbf':
    #     #     """keeping the sigma static to 1.0 currently"""
    #     #     sigma = 1.0
    #     #     pairwise_sq_dists = np.sum(X**2 , axis=1).reshape(-1,1) + np.sum(X**2 , axis=1) - 2 * np.dot(X , X.T)
    #     #     return np.exp(-pairwise_sq_dists / (2*sigma**2))
    #     return rbf_kernel(self.X , gamma=0.1)
    def compute_similarity_matrix(self):
        """Taking the pairwise maxtrix with euclidian distance"""
        X_pairwise_distances = pairwise_distances(self.X , metric="cosine")
        pairwise = X_pairwise_distances.max() - X_pairwise_distances
        return pairwise
        