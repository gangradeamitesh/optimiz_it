from .base_function import BaseFunction
import numpy as np
from sklearn.metrics import pairwise_distances


class GraphCut(BaseFunction):
    def __init__(self,X=None, selected_indices=None, gains=None, optimizer_type="naive_greedy") -> None:
        self.X = X
        self.simi_matrix = self.compute_similarity_matrix()
        self.n = self.simi_matrix.shape[0]
        self.d = self.simi_matrix.shape[1]
        self.selected_indices = selected_indices if selected_indices is not None else []
        self.current_value = np.zeros(self.d , dtype='float64')
        self.optimizer_type = optimizer_type
        self.ground_indices = [i for i in range(self.n)]
        super().__init__(selected_indices, gains, optimizer_type)

    def fit(self, subset_size):
        return super().fit(subset_size)

    def graph_cut(self,subset):
        print("Finding the gain for subset of indices:" , subset)
        """f_{gc}(X) = \\sum_{i \\in V, j \\in X} s_{ij} - \\lambda \\sum_{i, j \\in X} s_{ij}"""
        representation_term = sum(self.simi_matrix[i , j] for i in self.ground_indices for j in subset)
        # representation_term = 0
        # for i in self.ground_indices:
        #     for j in subset:
        #         print(self.simi_matrix[i,j])
        #         representation_term = representation_term + self.simi_matrix[i,j]
        diversity_term = sum(self.simi_matrix[i,j ] for i in subset for j in subset)
        # diversity_term = 0
        # for i in subset:
        #     for j in subset:
        #         print(self.simi_matrix[i,j])
        #         diversity_term = diversity_term +  self.simi_matrix[i,j]
        """Keeping the lambda as 0.4(static) """
        return representation_term - 0.1 * diversity_term

    def calculate_gain(self):
        gains = np.zeros(self.n , dtype='float64')
        for i in range(self.n):
            if i in self.selected_indices:
                continue
            print("Calculating gain for X data :" ,+self.X[i])
            gains[i] = self.graph_cut(list(self.selected_indices) + [i])
            print(gains[i])
        return gains

    def compute_similarity_matrix(self):
        """Taking the pairwise maxtrix with euclidian distance"""
        X_pairwise_distances = pairwise_distances(self.X , metric="euclidean")
        pairwise = X_pairwise_distances.max() - X_pairwise_distances
        return pairwise
