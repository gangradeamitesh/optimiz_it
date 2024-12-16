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
        self.selected_indices = selected_indices if selected_indices is not None else set()
        #self.gains = gains
        self.current_values = current_values if current_values is not None else np.zeros(self.d, dtype='float64')
        self.optimizer_type = optimizer_type
        super().__init__()
        

    
    def fit(self, subset_size):
        
        if subset_size==0 or subset_size==None:
            raise "Subset Size is either None or is Zero!"
        

        #validator.validate_simi_matrix()
        return super().fit(subset_size)
    
    def calculate_gain(self):
        gains = np.zeros(self.n , dtype = 'float64')
        for i in range(self.n):
            if i in self.selected_indices:
                continue
            gains[i] = np.maximum(self.simi_matrix[i] , self.current_values).sum()

        self.current_values = np.maximum(self.current_values , self.simi_matrix[np.argmax(gains)])
        
        # print("Gains---> " ,gains)
        # print("Simi Matrix -->  " , self.simi_matrix)
        # print("selected indices ->" , self.selected_indices)
        # print("Current Values ->" , self.current_values)
        return gains
        #return calculate_gain_facility(self.simi_matrix ,self.n, self.selected_indices , self.current_values)

    
    def compute_similarity_matrix(self):
        """Taking the pairwise maxtrix with euclidian distance"""
        X_pairwise_distances = pairwise_distances(self.X , metric="cosine")
        pairwise = X_pairwise_distances.max() - X_pairwise_distances
        return pairwise