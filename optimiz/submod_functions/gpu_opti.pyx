import torch
import numpy as np
cimport numpy as cnp
cimport torch

def calculate_gain_facility(cnp.ndarray[cnp.float64_t , ndim=1] gains , cnp.ndarray[cnp.float64_t , ndim=2] simi_matrix , int n  , cnp.ndarray[cnp.npy_long , ndim=1] selected_indices , cnp.ndarray[cnp.float64_t, ndim=1] current_values):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        #print("Using MPS")
    else:
        device = "cpu"
    
    gains_tensor = torch.tensor(gains , device=device , dtype=torch.float32)
    simi_matrix_tensor = torch.tensor(simi_matrix , device=device , dtype=torch.float32)
    selected_indices_tensor = torch.tensor(selected_indices , device=device , dtype=torch.float32)
    current_values_tensor = torch.tensor(current_values , device=device , dtype=torch.float32)

    for i in range(n):
        if i in selected_indices_tensor:
            continue
        gains_tensor[i] = torch.maximum(simi_matrix_tensor[i], current_values_tensor).sum()
    return gains_tensor.cpu().numpy()