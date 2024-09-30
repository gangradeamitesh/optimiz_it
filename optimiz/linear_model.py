import numpy as np
from .gradient_descent import GradientDescent
from .utils import Preprocessing
from .optimizer_factory import OptimizerFactory
from .losses import mse_loss

class LinearRegression:

    def __init__(self, learning_rate = 0.01 , iterations = 1000 , tolerance = 1e-6 , optimizer_type) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tolerance = tolerance
        self.weights = None
        self.optimizer_type = optimizer_type,
        self.optimizer = OptimizerFactory.get_optimizer(
            optimizer_type , 
            learning_rate , 
            iterations,
            tolerance
        )
        self.preprocessing = Preprocessing()
    
    def fit(self, X , y , scale = False):

        X_with_bias = np.c_[np.ones((X.shape[0] , 1)) , X]
        print(f"Value of Scale {scale}")

        if scale:
            X = self.preprocessing.scale(X)
        

        initial_weights = np.zeros(X_with_bias.shape[1])

        self.weights = self.optimizer.optimize(X_with_bias , y , initial_weights)
    
    def predict(self , X):

        if self.weights is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        X_with_bias = np.c_[np.ones((X.shape[0] , 1 )) , X]
        return X_with_bias.dot(self.weights)
    
    def mse_score(self , X , y):

        y_pred = self.predict(X)
        mse = mse_loss(y_true= y , y_pred= y_pred)
        return mse
 