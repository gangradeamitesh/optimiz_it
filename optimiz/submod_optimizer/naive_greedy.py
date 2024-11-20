from .base_optimizer import SubModOptimizer
import numpy as np

class NaiveGreedy(SubModOptimizer):
    def __init__(self, function=None, gains=None) -> None:
        super().__init__(function, gains)
        
    def select(self, subset_size , function):
        for _ in range(subset_size):
            #print(function)
            gains = function.calculate_gain()
            best_item = np.argmax(gains)
            function.selected_indices.add(best_item)
            function.current_values = np.maximum(function.current_values , function.simi_matrix[best_item])
        #print(function.selected_indices)
        return function.selected_indices
