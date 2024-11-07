
from base_classifier import BaseClassifier
from optimizer_factory import OptimizerFactory
from .preprocessing import Preprocessing
import numpy as np



class LinearClassifier(BaseClassifier):
    def __init__(self , tolerance, learning_rate , iterations , optimizer_type = None , method = None) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.optimizer = OptimizerFactory.get_optimizer(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            iterations=iterations,
            tolerance=tolerance,
            method = method,
        )
        self.preprocessing = Preprocessing()

    def fit(self, X, y , scale = False):
        if scale:
            X = self.preprocessing.scale(X)
        X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
        initial_weights = np.zeros(X_with_bias.shape[1])
        y = y.flatten()
        
        self.weights , loss_history = self.optimizer.optimize(X_with_bias, y, initial_weights)
        return loss_history        

    def predict(self, X):
        return None