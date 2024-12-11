import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from optimiz.classifier import LinearClassifier
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
from optimiz.facility_location import FacilityLocation

# device = torch.device('mps') if torch.has_mps else torch.device('cpu')

# print("Devicce :", device)

from sklearn.datasets import make_classification
from sklearn.datasets import fetch_openml

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, random_state=42)



print("Size of X :" , X.shape)
print("Size of y : " , y.shape)

# facilityLocation = FacilityLocation(X , y)
# #print(facilityLocation.idx)
# subset = facilityLocation.greedy_selection(subset_size=2000)
# print(len(subset))

#split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                 test_size = 0.3,
                                                 random_state = 42)

clf = LinearClassifier(learning_rate=0.00001 ,tolerance=0.000001, iterations=1000 , optimizer_type="newton")
clf.fit(X_train , y_train)