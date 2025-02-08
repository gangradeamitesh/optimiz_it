import numpy as np 
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from optimiz.submod_functions.facility_location import FacilityLocation

np.set_printoptions(suppress=True)


def generate_blob(n_samples=1000 , n_features=2 , n_centers =5 , random_state=42):
    num_clusters = 10
    cluster_std_dev = 4
    X , y = make_blobs(n_samples=500, centers=num_clusters, 
                                          n_features=2, cluster_std=cluster_std_dev, center_box=(0,100), 
                                          return_centers=True, random_state=4)
    # X , y = make_blobs(n_samples=n_samples ,centers=n_centers , n_features=n_features,cluster_std=0.5, random_state=42)
    return X , y

def visualize_subset(X , y , selected_indicees):
    X_subset = X[selected_indicees]
    plt.figure(figsize=(8,6))
    plt.scatter(X[: , 0] , X[:,1] ,c = y , cmap="viridis" , alpha=0.6 , label="All points")
    plt.scatter(X_subset[: , 0] , X_subset[: ,1], c = 'red' , edgecolor='k', s=100, label="Selected Subset")
    plt.legend()
    plt.show()

if __name__=="__main__":
    X , y = generate_blob()
    #X = np.array([1, 2, 3, 5, 6, 7]).reshape(-1, 1)
    print("X :" , X)
    print("Shape of data X :" , X.shape)
    facilityLoc = FacilityLocation(X=X)
    print("Similarity Matrix Shape", facilityLoc.compute_similarity_matrix())
    subset = facilityLoc.fit(subset_size=10)
    print(subset)
    print('Subset Size : ' , len(subset))
    #print(X[subset])
    visualize_subset(X , y , list(subset))


















# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import time
# #from optimiz.classifier import LinearClassifier
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# import torch
# from optimiz.submod_functions.facility_location import FacilityLocation
# from sklearn.datasets import make_classification
# from sklearn.datasets import fetch_openml
# from sklearn.linear_model import LogisticRegression

# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import make_blobs
# from sklearn.preprocessing import StandardScaler
# # X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42,)
# # X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

# X , y = make_blobs(n_samples=10000 , centers=5 ,n_features=50, cluster_std=0.5 , random_state=42)

# # print("X shape : " , X.shape)
# # scaler = StandardScaler()
# # X = scaler.fit_transform(X)

# facilityLocation = FacilityLocation(X=X)
# #print(facilityLocation.idx)
# subset = facilityLocation.fit(subset_size=10)
# print("Length of Subset " , len(subset))
# print(subset)

# X_subset = X[list(subset)]
# plt.figure(figsize=(10,10))
# plt.scatter(X[: , 0] , X[:,1] , c = y , cmap="viridis" , s=50 , alpha=0.7)
# plt.scatter(X_subset[:,0] , X_subset[:,1] , color = 'red' , s = 100, label="Subset Data")
# # plt.scatter(X, np.zeros_like(X), c=y, s=10, cmap='viridis', alpha=0.5)  # Use a constant y-value for 1D
# # plt.scatter(X_subset, np.zeros_like(X_subset), color = 'red',label="Subset Data")  # Use a constant y-value for 1D


# plt.title("Visualization of Generated Blobs")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.colorbar(label='Cluster Label')
# plt.grid()
# plt.show()

# #print(list(subset))


