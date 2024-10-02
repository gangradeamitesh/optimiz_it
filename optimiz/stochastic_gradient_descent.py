from .optimizer import Optimizier
import random
import numpy as np
from .gradient_descent import GradientDescent
from .losses import mse_loss

class StocasticGradientDescent(Optimizier):

    def __init__(self, learning_rate, iterations, tolerance , batch_size=1) -> None:
        super().__init__(learning_rate, iterations, tolerance)
        self.gradient_descent = GradientDescent(learning_rate=learning_rate, iterations=iterations , tolerance=tolerance)
        self.batch_size = batch_size
    
    def optimize(self , X , y , initial_weights=None , batch_size = 1):
        
        m, n = X.shape
        weights = initial_weights.copy()

        for i in range(1 , self.iterations+1):
            idxs = np.arange(m)
            #shuffling the index
            np.random.shuffle(idxs)
            X = X[idxs]
            y = y[idxs]

            for j in range(0 , m , batch_size):
                X_batch = X[j : j + batch_size]
                y_batch = y[j : j + batch_size]
                gradient = self.gradient_descent._compute_gradient(X_batch , y_batch , weights)
                weights -= self.learning_rate * gradient
            if i % 100 == 0:
                print(f"Iteration : {i} , Loss : {mse_loss(y , X.dot(weights))}")
            
        return weights


    
    # def _compute_gradient(self, X, y, weights):
    #     """
    #     Computes the gradient for the given weights.

    #     Args:
    #         X (numpy.ndarray): The input feature matrix.
    #         y (numpy.ndarray): The target values.
    #         weights (numpy.ndarray): The current weights.

    #     Returns:
    #         numpy.ndarray: The computed gradient.
    #     """
    #     m = len(y)
    #     predictions = X.dot(weights)
    #     gradient = (1 / m) * X.T.dot(predictions - y)
    #     return gradient