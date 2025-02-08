from abc import ABC, abstractmethod
import numpy as np

class BaseClassifier(ABC):

    def __init__(self) -> None:
        self.weights = None

    @abstractmethod
    def fit(self , X , y):
        pass

    @abstractmethod
    def predict(self , X):
        pass

    