import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from optimiz.classifier import LinearClassifier

from sklearn.datasets import load_wine

data = load_wine()
#convert to a dataframe
df = pd.DataFrame(data.data, columns = data.feature_names)
#create the species column
df['Class'] = data.target
#replace this with the actual names
target = np.unique(data.target)
target_names = np.unique(data.target_names)
targets = dict(zip(target, target_names))
df['Class'] = df['Class'].replace(targets)

#extract features and target variables
x = df.drop(columns="Class")
y = df["Class"]
#save the feature name and target variables
feature_names = x.columns
labels = y.unique()
#split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,
                                                 test_size = 0.3,
                                                 random_state = 42)

clf = classifier()