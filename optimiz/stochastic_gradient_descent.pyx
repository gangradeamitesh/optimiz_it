from .optimizer import Optimizier
import random
import numpy as np
cimport numpy as cnp
from .gradient_descent import GradientDescent
from .losses import mse_loss
from .utils import _compute_gradient

class StocasticGradientDescent(Optimizier):

    def __init__(self, learning_rate, iterations, tolerance , batch_size=1) -> None:
        super().__init__(learning_rate, iterations, tolerance)
        self.batch_size = batch_size
    
    def optimize(self , cnp.ndarray[cnp.float64_t , ndim=2] X ,cnp.ndarray[cnp.float64_t , ndim=1] y ,cnp.ndarray[cnp.float64_t , ndim=1] initial_weights):
        cdef int i , j
        cdef cnp.ndarray[cnp.float64_t , ndim=1] weights
        cdef cnp.ndarray[cnp.float64_t , ndim=2] X_batch 
        cdef cnp.ndarray[cnp.float64_t , ndim=1] y_batch
        cdef list loss_history = []
        cdef int m = X.shape[0]
        cdef int n = X.shape[1]

        weights = initial_weights.copy()
        for i in range(1 , self.iterations+1):
            #shuffling the index

            for j in range(0 , m , self.batch_size):
                X_batch = X[j : j + self.batch_size]
                y_batch = y[j : j + self. batch_size]
                gradient = self._compute_gradient(X_batch , y_batch , weights)
                weights -= self.learning_rate * gradient
            loss = mse_loss(y ,np.dot(X , weights))
            loss_history.append(loss)
            if i % 100 == 0:
                print(f"Iteration : {i} , Loss : {mse_loss(y , X.dot(weights))}")
            
        return weights , loss_history
    def _compute_gradient(self, cnp.ndarray[cnp.float64_t , ndim=2 ] X , cnp.ndarray[cnp.float64_t , ndim=1] y , cnp.ndarray[cnp.float64_t , ndim=1] weights):
        cdef int m = len(y)
        cdef cnp.ndarray[cnp.float64_t] predictions = X.dot(weights)
        cdef cnp.ndarray[cnp.float64_t] gradient = (1 / m) * X.T.dot(predictions - y)
        return gradient