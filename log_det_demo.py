import numpy as np 
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from optimiz.submod_functions.log_det import LogDeterminant
np.set_printoptions(suppress=True)


def generate_blob(n_samples=10  , n_features=2 , n_centers =5 , random_state=42):

    X , y = make_blobs(n_samples=n_samples ,centers=n_centers , n_features=n_features,cluster_std=0.5, random_state=42)
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
    print("Shape of data X :" , X.shape)
    logDet = LogDeterminant(X=X)
    print("Similarity Matrix Shape", logDet.compute_similarity_matrix())
    subset = logDet.fit(subset_size=5)
    print(subset)
    print('Subset Size : ' , len(subset))
    visualize_subset(X , y , list(subset))
