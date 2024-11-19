from .base_optimizer import SubModOptimizer
import numpy as np

class NaiveGreedy(SubModOptimizer):
    def __init__(self, function=None, gains=None) -> None:
        super().__init__(function, gains)
        
    def select(self, subset_size):
        for _ in range(subset_size):

            gains = self.function.calculate_gain()
            best_item = np.argmax(gains)
            self.function.selected_indices.add(best_item)
            self.function.current_valuers = np.maxinum(self.function.current_values , self.function.simi_matrix[best_item])
