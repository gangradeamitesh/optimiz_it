import numpy as np
cimport numpy as cnp
from .optimizer import Optimizier
from .losses import mse_loss

class GradientDescent(Optimizier):

    def __init__(self, learning_rate=None, iterations=None, tolerance=None) -> None:
        super().__init__(learning_rate, iterations, tolerance)
    
    def optimize(self, cnp.ndarray[cnp.float64_t, ndim=2] X , cnp.ndarray[cnp.float64_t , ndim=1] y , cnp.ndarray[cnp.float64_t , ndim=1] initial_weights):
        cdef int i
        cdef int m = X.shape[0]
        cdef int n = X.shape[1]
        cdef cnp.ndarray[cnp.float64_t, ndim=1] weights
        cdef list loss_history = []

        weights = initial_weights.copy()

        for i in range(1 , self.iterations):
            gradient = self._compute_gradient(X , y , weights)
            weights -= self.learning_rate * gradient
            loss_history.append(mse_loss(y , np.dot(X , weights)))
            if i % 100 == 0:
                print(f"Iteration : {i} , Loss : {loss_history[i]}")
        return weights , loss_history
    
    def _ompute_gradient(self , cnp.ndarray[cnp.float64_t , ndim=2] X, cnp.ndarray[cnp.float64_t , ndim=1] y, cnp.ndarray[cnp.float64_t , ndim=1] weights):
        m = len(y)
        predictions = X.dot(weights)
        gradient = (1 / m) * X.T.dot(predictions - y)
        return gradient


        return None
