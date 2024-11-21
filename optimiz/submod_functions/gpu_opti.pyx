import torch
import numpy as np
cimport numpy as cnp
cimport torch

def calculate_gain_facility(cnp.ndarray[cnp.float64_t , ndim=2] simi_matrix , int n  , 
    object selected_indices ,
    cnp.ndarray[cnp.float64_t, ndim=1] current_values):

    # if torch.cuda.is_available():
    #     device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    #     #print("Using MPS")
    # else:
    #     device = "cpu"
    
    #gains_tensor = torch.tensor(gains , device=device , dtype=torch.float32)
    # simi_matrix_tensor = torch.tensor(simi_matrix , device=device , dtype=torch.float32)
    # selected_indices_tensor = torch.tensor(selected_indices , device=device , dtype=torch.float32)
    # current_values_tensor = torch.tensor(current_values , device=device , dtype=torch.float32)

    # for i in range(n):
    #     if i in selected_indices_tensor:
    #         continue
    #     gains_tensor[i] = torch.maximum(simi_matrix_tensor[i], current_values_tensor).sum()
    # return gains_tensor.cpu().numpy()
    cdef:
        cnp.ndarray[double , ndim=1] gains = np.zeros(n , dtype=np.float64)
        cnp.npy_long i , j
        double max_val , total
        #set selected_indices = set(selected_indices)    
    
    for i in range(n):
        if i in selected_indices:
            continue
        total = 0.0
        for j in range(current_values.shape[0]):
            max_val = simi_matrix[i , j] if simi_matrix[i , j] > current_values[j] else current_values[j]
            total += max_val
        gains[i] = total
    return gains