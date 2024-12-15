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
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
# X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42,)
# X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

X , y = make_blobs(n_samples=10000 , centers=3 ,n_features=5, cluster_std=0.5 , random_state=42)

print("X shape : " , X.shape)
scaler = StandardScaler()
X = scaler.fit_transform(X)

facilityLocation = FacilityLocation(X=X)
#print(facilityLocation.idx)
subset = facilityLocation.fit(subset_size=3)
print("Length of Subset " , len(subset))
print(subset)

X_subset = X[list(subset)]
plt.figure(figsize=(10,10))
plt.scatter(X[: , 0] , X[:,1] , c = y , cmap="viridis" , s=50 , alpha=0.7)
plt.scatter(X_subset[:,0] , X_subset[:,1] , color = 'red' , s = 100, label="Subset Data")

plt.title("Visualization of Generated Blobs")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label='Cluster Label')
plt.grid()
plt.show()

#print(list(subset))