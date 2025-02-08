from .base_function import BaseFunction
from optimiz import validator
from sklearn.metrics import pairwise_distances
import numpy as np
from .gpu_opti import calculate_gain_facility


class FacilityLocation(BaseFunction):

    def __init__(self,X=None,simi_matrix=None, gains=None, current_values=None, selected_indices=None , optimizer_type="naive_greedy") -> None:
        self.X = X
        self.simi_matrix = self.compute_similarity_matrix()
        self.n = self.simi_matrix.shape[0]
        self.d = self.simi_matrix.shape[1]
        self.selected_indices = selected_indices if selected_indices is not None else []
        #self.gains = gains
        self.current_values = 0
        self.optimizer_type = optimizer_type
        self.ground_indices = [i for i in range(self.n)]

        super().__init__()
    
    def fit(self, subset_size):
        
        if subset_size==0 or subset_size==None:
            raise "Subset Size is either None or is Zero!"

        return super().fit(subset_size)
    
    def gain(self,subset):
        print("Finding the gain for subset of indices:" , subset)
        """f(X) = \\sum_{i \\in V} \\max_{j \\in X} s_{ij}"""
        if len(subset)==0:
            return 0
        #representation_term = sum(max(self.simi_matrix[i,j] for j in subset) for i in self.ground_indices)
        representation_term = np.sum(np.max(self.simi_matrix[:, subset], axis=1))
        return representation_term
    
    def calculate_gain(self):
        gains = np.zeros(self.n , dtype = 'float64')
        for i in range(self.n):
            if i in self.selected_indices:
                continue
            gains[i] = self.gain(self.selected_indices+[i]) - self.current_values
        return gains 

    
    def compute_similarity_matrix(self):
        """Taking the pairwise maxtrix with euclidian distance"""
        X_pairwise_distances = pairwise_distances(self.X , metric="euclidean")
        return X_pairwise_distances.max() - X_pairwise_distances
        