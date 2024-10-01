import numpy as np
from .optimizer import Optimizier
from .losses import mse_partial_gradient, mse_loss

class CoordinateDescent(Optimizier):
    """
    Coordinate Descent optimization algorithm.

    Args:
        learning_rate (float): The step size for weight updates.
        iterations (int): The number of iterations to perform.
        tolerance (float): The tolerance for stopping criteria.
    """

    def __init__(self, learning_rate, iterations, tolerance) -> None:
        """
        Initializes the CoordinateDescent optimizer.

        Args:
            learning_rate (float): The step size for weight updates.
            iterations (int): The number of iterations to perform.
            tolerance (float): The tolerance for stopping criteria.
        """
        super().__init__(learning_rate, iterations, tolerance)

    def optimize(self, X, y, initial_weights=None):
        """
        Performs the optimization process using Coordinate Descent.

        Args:
            X (numpy.ndarray): The input feature matrix.
            y (numpy.ndarray): The target values.
            initial_weights (numpy.ndarray, optional): Initial weights for optimization.

        Returns:
            numpy.ndarray: The optimized weights after the process.
        """
        m, n = X.shape
        weights = initial_weights.copy()

        for i in range(self.iterations):
            for j in range(n):
                partial_j_gradient = mse_partial_gradient(X, y, weights, j)
                weights[j] = weights[j] - self.learning_rate * partial_j_gradient

            if i % 100 == 0:
                print(f"Iteration : {i} , Loss : {mse_loss(y, X.dot(weights))} ")

        return weights