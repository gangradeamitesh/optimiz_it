import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from optimiz.classifier import LinearClassifier
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
from optimiz.submod_functions.facility_location import FacilityLocation
from sklearn.datasets import make_classification
from sklearn.datasets import fetch_openml

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, random_state=42,)

facilityLocation = FacilityLocation(X=X)
#print(facilityLocation.idx)
subset = facilityLocation.fit(subset_size=1000)
print(subset)