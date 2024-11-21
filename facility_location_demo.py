import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
#from optimiz.classifier import LinearClassifier
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
from optimiz.submod_functions.facility_location import FacilityLocation
from sklearn.datasets import make_classification
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=6000, n_features=20, n_classes=2, random_state=42,)
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

start_time = time.time()
facilityLocation = FacilityLocation(X=X)
#print(facilityLocation.idx)
subset = facilityLocation.fit(subset_size=600)
end_time = time.time()

print(end_time - start_time)


start_time_2 = time.time()
clf = LogisticRegression(random_state = 42)
clf.fit(X , y)
end_time_2 = time.time()
print(end_time_2 - start_time_2)

start_time_3 = time.time()
clf_2 = LogisticRegression(random_state = 42)
clf_2.fit(X[list(subset)] , y[list(subset)])
end_time_3 = time.time()
print(end_time_3 - start_time_3)