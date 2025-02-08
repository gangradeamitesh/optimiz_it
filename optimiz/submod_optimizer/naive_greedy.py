from .base_optimizer import SubModOptimizer
import numpy as np

class NaiveGreedy(SubModOptimizer):
    def __init__(self, function=None, gains=None) -> None:
        super().__init__(function, gains)
        
    def select(self, subset_size , function):
        for _ in range(subset_size):
            #print(function)

            #print("Calling Function Calculate Gain")
            gains = function.calculate_gain()
            #print(gains)
            best_item = np.argmax(gains)
            function.selected_indices.append(int(best_item))
            #print("selected indices " , function.selected_indices)
            self.current_value = function.gain(function.selected_indices)
            #print("calculated current value for : ", function.selected_indices)
            #print(best_item)

        #print(function.selected_indices)
        return function.selected_indices
