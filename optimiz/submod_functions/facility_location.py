from .base_function import BaseFunction
from optimiz import validator
from sklearn.metrics import pairwise_distances
import numpy as np

class FacilityLocation(BaseFunction):

    def __init__(self,X=None,simi_matrix=None, gains=None, current_values=None, selected_indices=None , optimizer_type="naive_greedy") -> None:
        self.X = X
        self.simi_matrix = self.compute_similarity_matrix()
        self.n = self.simi_matrix.shape[0]
        self.d = self.simi_matrix.shape[1]
        self.selected_indices = selected_indices if selected_indices is not None else set()
        #self.optimizer_type = optimizer_type
        self.gains = gains
        self.current_values = current_values if current_values is not None else np.zeros(self.d, dtype='float64')
        self.optimizer_type = optimizer_type
        super().__init__(gains= gains)
        

    
    def fit(self, subset_size):
        
        if subset_size==0 or subset_size==None:
            raise "Subset Size is either None or is Zero!"
        

        #validator.validate_simi_matrix()
        return super().fit(subset_size)
    
    def calculate_gain(self):
        print(self.n)
        gains = np.zeros(self.n , dtype = 'float64')
        for i in range(self.n):
            if i in self.selected_indices:
                continue
            print(gains[i].shape)
            print(self.simi_matrix[i].shape)
            print(self.current_values.shape)
            #print("Max of simi and curr :  ", np.maximum(self.simi_matrix[i] , self.current_values))
            gains[i] = np.maximum(self.simi_matrix[i] , self.current_values).sum()
            print(gains[i])
        
        return gains
    
    
    def compute_similarity_matrix(self):
        """Taking the pairwise maxtrix with euclidian distance"""
        X_pairwise_distances = pairwise_distances(self.X , metric="euclidean" , squared=True)
        pairwise = X_pairwise_distances.max() - X_pairwise_distances
        return pairwise