import numpy as np
from .optimizer import Optimizier
from .losses import mse_partial_gradient , mse_loss

class CoordinateDescent(Optimizier):

    def __init__(self, learning_rate, iterations, tolerance) -> None:
        super().__init__(learning_rate, iterations, tolerance)
    
    def optimize(self , X , y , initial_weights = None):

        m , n = X.shape

        for i in range(self.iterations):
            for j in range(n):
                partial_j_gradient = mse_partial_gradient(X , y , initial_weights , j)
                initial_weights[j] = initial_weights[j] -  self.learning_rate * partial_j_gradient

        return initial_weights
    