cimport numpy as cnp
from .optimizer import Optimizier

class GradientDescent(Optimizier):

    def __init__(self, learning_rate=None, iterations=None, tolerance=None) -> None:
        super().__init__(learning_rate, iterations, tolerance)
    
    def optimize(self , ctypedef cnp.ndarray(cnp.float64_t , ndim=1) X):
