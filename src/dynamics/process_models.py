import numpy as np

def decoupled_CV_model(state, T):
    x = state.copy()
    kinematic_state = x[:6]

    t = np.array([[1, T],
                  [0, 1]])
    F = np.kron(t, np.eye(3))

    x[:6] = F @ kinematic_state
    
    return x

def decoupled_CV_model_jacobian(state, T, N_pca):
    kinematic_state = state[:6]

    t = np.array([[1, T],
                  [0, 1]])
    F1 = np.kron(t, np.eye(3))

    F = np.block([[F1, np.zeros((6, 2+N_pca))],
                  [np.zeros((2+N_pca, 6)), np.eye(2+N_pca)]])

    return F
