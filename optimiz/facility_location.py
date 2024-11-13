import numpy as np
from sklearn.metrics import pairwise_distances

class FacilityLocation:

    def __init__(self , X , y) -> None:
        self.X = X
        self.y = y
        
        self.X_simi = self.compute_similarity_matrix()
        self.n = self.X_simi.shape[0]
        self.d = self.X_simi.shape[1]
        self.ranking = None
        #self.gains = []
        self.current_values = np.zeros(self.d, dtype='float64')
        #self.current_concave_values = np.zeros(self.d , dtype='float64')
        self.idx = np.zeros(self.n, dtype='int8')
        self.selected_indices = set()
        
    def compute_similarity_matrix(self):
        """Taking the pairwise maxtrix with euclidian distance"""
        X_pairwise_distances = pairwise_distances(self.X , metric="euclidean" , squared=True)
        #print("Calculating paiwise distance")
        pairwise = X_pairwise_distances.max() - X_pairwise_distances
        #print("Done calculating pairwise distance")
        print(pairwise.ndim)
        return pairwise
    def calculate_gains(self):
        gains = np.zeros(self.n, dtype='float64')
        for i in range(self.n):
            if i in self.selected_indices:
                #print(f"{i} already selected ----------------------------------")
                continue
            gains[i] = np.maximum(self.X_simi[i], self.current_values).sum()
            #if (i+1)%5000==0:
                #print(f"Done calculating gain for {i+1}") 
        return gains
    
    def greedy_selection(self , subset_size):
        for _ in range(subset_size):
            #if (_+1)%100==0:
                #print(f"On the subset number {_+1}")
            
            gains = self.calculate_gains()
            
            best_item = np.argmax(gains)
            self.selected_indices.add(best_item)

            self.current_values = np.maximum(self.current_values , self.X_simi[best_item])

        return self.selected_indices
    