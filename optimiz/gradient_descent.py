from .optimizer import Optimizier
import numpy as np
from .utils import _compute_gradient
#from losses import mse_loss

class GradientDescent(Optimizier):
    """
    Gradient Descent optimization algorithm.

    Args:
        learning_rate (float): The step size for weight updates.
        iterations (int): The number of iterations to perform.
        tolerance (float): The tolerance for stopping criteria.
    """

    def __init__(self, learning_rate, iterations, tolerance) -> None:
        """
        Initializes the GradientDescent optimizer.

        Args:
            learning_rate (float): The step size for weight updates.
            iterations (int): The number of iterations to perform.
            tolerance (float): The tolerance for stopping criteria.
        """
        super().__init__(learning_rate, iterations, tolerance)

    def optimize(self, X, y, initial_weights=None):
        """
        Performs the optimization process using Gradient Descent.

        Args:
            X (numpy.ndarray): The input feature matrix.
            y (numpy.ndarray): The target values.
            initial_weights (numpy.ndarray, optional): Initial weights for optimization.

        Returns:
            numpy.ndarray: The optimized weights after the process.
        """
        m, n = X.shape
        weights = initial_weights.copy()
        loss_history = []
        for i in range(self.iterations):
            gradient = _compute_gradient(X, y, weights)
            updated_weights = weights - self.learning_rate * gradient
            weights = updated_weights
            loss = self.mse_loss(y , X.dot(weights))
            if i % 100 == 0:
                print(f"Iteration : {i} , Loss : {loss} ")
            loss_history.append(loss)


        return weights , loss_history

    def _compute_cost(self, X, y, weights):
        """
        Computes the cost for the given weights.

        Args:
            X (numpy.ndarray): The input feature matrix.
            y (numpy.ndarray): The target values.
            weights (numpy.ndarray): The current weights.

        Returns:
            float: The computed cost.
        """
        m = len(y)
        predictions = X.dot(weights)
        return (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    def mse_loss(self , y_true, y_pred):
        mse = (1 / 2) * np.mean((y_true - y_pred) ** 2)
        return mse

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