import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2, gamma
from typing import Dict, List, Union, Tuple

# Initialize project and import modules
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir
))
sys.path.append(PROJECT_ROOT)

from utils.tools import ssa, StateIdxToName
from src.utils.SimulationResult import SimulationResult

def calculate_nees_for_components_over_time(
    state_estimates_list: List[np.ndarray],
    state_gt_list: List[np.ndarray],
    covariance_list: List[np.ndarray],
    nees_config: Dict[str, Union[List[int], Tuple[int, int], slice]]
) -> Dict[str, np.ndarray]:
    """
    Calculates NEES for multiple components over a sequence of time steps.

    Args:
        state_estimates_list (List[np.ndarray]): List of state estimates for each time step.
        state_gt_list (List[np.ndarray]): List of ground truth states for each time step.
        covariance_list (List[np.ndarray]): List of covariance matrices for each time step.
        nees_config (Dict): A dictionary defining the components to calculate NEES for.
                            Keys are the names of the components (e.g., "position").
                            Values are lists of indices, tuples for ranges, or slices
                            corresponding to that component in the state vector.
                            Example:
                            {
                                "all": "all",
                                "position": [0, 1],
                                "heading": [2],
                                "pca_coeffs": slice(8, None)
                            }

    Returns:
        Dict[str, np.ndarray]: A dictionary where keys are component names and values
                               are numpy arrays of NEES values over time for that component.
    """
    num_timesteps = len(state_estimates_list)
    
    # Initialize a dictionary to hold the time series of NEES for each component
    nees_results_over_time = {name: [] for name in nees_config.keys()}

    for t in range(num_timesteps):
        state_estimate = state_estimates_list[t]
        state_gt = state_gt_list[t]
        covariance = covariance_list[t]

        assert state_estimate.ndim == 1, f"state_estimate at t={t} must be a 1D vector"
        assert state_gt.ndim == 1, f"state_gt at t={t} must be a 1D vector"
        assert covariance.shape == (state_estimate.shape[0], state_estimate.shape[0]), \
            f"Covariance shape at t={t} is not compatible with state shape"

        error = state_estimate - state_gt
        if error.shape[0] > 2:
            error[2] = ssa(error[2])

        for name, indices in nees_config.items():
            nees_val = 0.0
            try:
                if isinstance(indices, str) and indices == "all":
                    sub_error, sub_cov = error, covariance
                else:
                    sub_error = error[indices]
                    sub_cov = covariance[np.ix_(indices, indices)]

                if sub_error.ndim == 0 or len(sub_error) == 1:  # Scalar case
                    scalar_error = sub_error.item() if isinstance(sub_error, np.ndarray) else sub_error
                    scalar_cov = sub_cov.item() if isinstance(sub_cov, np.ndarray) else sub_cov
                    if scalar_cov > 1e-9:
                        nees_val = scalar_error**2 / scalar_cov
                    else:
                        print(f"Warning at t={t} for '{name}': Covariance too small ({scalar_cov}). Defaulting NEES to 1.0.")
                        nees_val = 1.0
                else:  # Vector case
                    nees_val = sub_error.T @ np.linalg.solve(sub_cov, sub_error)

            except (np.linalg.LinAlgError, ZeroDivisionError) as e:
                print(f"Warning at t={t} for '{name}': {e}. Defaulting NEES to 1.0.")
                nees_val = 1.0

            if nees_val < 0:
                print(f"\n--- Negative NEES Debug Info at t={t} ---")
                print(f"Component: '{name}' (Indices: {indices})")
                print(f"Negative NEES value: {nees_val}")
                print(f"Sub-state error (sub_error): {sub_error}")
                print(f"Sub-covariance (sub_cov):\n{sub_cov}")
                try:
                    # Condition number indicator of a matrix being close to singular
                    cond_num = np.linalg.cond(sub_cov)
                    print(f"Condition number of sub_cov: {cond_num:.2e}")
                except np.linalg.LinAlgError:
                    print("Could not compute condition number for sub_cov.")
                print("--- End Debug Info ---\n")
            nees_results_over_time[name].append(max(0.0, nees_val)) # Ensure non-negative
            
    # Convert lists to numpy arrays for easier plotting and analysis
    for name, values in nees_results_over_time.items():
        nees_results_over_time[name] = np.array(values)
        
    return nees_results_over_time

def calculate_nis_over_time(
    innovations_list: List[np.ndarray],
    innovation_covariances_list: List[np.ndarray]
) -> np.ndarray:
    """
    Calculates the Normalized Innovation Squared (NIS) over a sequence of time steps.

    Args:
        innovations_list (List[np.ndarray]): List of innovation vectors (y) for each time step.
        innovation_covariances_list (List[np.ndarray]): List of innovation covariance matrices (S)
                                                       for each time step.

    Returns:
        np.ndarray: A numpy array of NIS values over time.
    """
    num_timesteps = len(innovations_list)
    assert num_timesteps == len(innovation_covariances_list), "Input lists must have the same length."

    nis_results_over_time = []

    for t in range(num_timesteps):
        innovation = innovations_list[t]
        s_matrix = innovation_covariances_list[t]
        nis_val = 0.0

        # Ensure innovation is a 1D vector
        if innovation.ndim > 1:
            innovation = innovation.flatten()

        assert innovation.shape[0] == s_matrix.shape[0], \
            f"Shape mismatch at t={t}: innovation has {innovation.shape[0]} elements, S has shape {s_matrix.shape}"

        try:
            # Calculate NIS: v.T * S^-1 * v
            nis_val = innovation.T @ np.linalg.solve(s_matrix, innovation)

        except np.linalg.LinAlgError as e:
            print(f"Warning for NIS at t={t}: Could not solve S matrix. {e}. Defaulting NIS to 1.0.")
            nis_val = 1.0

        if nis_val < 0:
            print(f"Warning for NIS at t={t}: Negative NIS value ({nis_val}). Clamping to 0.")
            nis_val = 0.0
        
        nis_results_over_time.append(nis_val)

    return np.array(nis_results_over_time)


def calculate_NEES_single_sim(state_gt, state_estimates, covariances):
    nees = []
    for t in range(len(state_estimates)):
        error = state_estimates[t] - state_gt[t]
        error[2] = ssa(error[2])  # Normalize angle error
        P = covariances[t]
        nees_t = (error.T @ np.linalg.inv(P) @ error).item()
        nees.append(nees_t)
    return nees

def calculate_NIS_single_sim(y_measurements, S_matrices):
    nis = []
    for t in range(len(y_measurements)):
        y = y_measurements[t]


        v = y - z[t] # Innovation



        S = S_matrices[t]
        nis_t = (v.T @ np.linalg.inv(S) @ v).item()
        nis.append(nis_t)
    return nis

NAME = 'bfgs'

PLOT_SAVE_PATH = f'figures/consistency_{NAME}.svg'

FIGURE_TITLE = f'Consistency Analysis: {NAME}'


sim_data_path = f'data/results/simulation_data_martin_sim_bfgs_ellipse_50frames.pkl'

# Load the simulation data
sim_data = np.load(sim_data_path, allow_pickle=True)

# Extract the data
state_estimates = sim_data['state_estimates']
ground_truth = sim_data['ground_truth']
static_covariances = sim_data['static_covariances']
true_extent = sim_data['true_extent']
P_prior = sim_data['P_prior']
P_post = sim_data['P_post']
print ("P_prior shape: ", np.array(P_prior).shape)
print ("P_post shape: ", np.array(P_post).shape)

# print("P_post: ", P_post)
S = sim_data['S']
# print ("S shape: ", np.array(S).shape)
# print ("S : ", S)
y = sim_data['y']
z = sim_data['z']
x_dim = sim_data['x_dim']
z_dim = sim_data['z_dim']

M = sim_data['PCA_eigenvectors']
fourier_coeff_mean = sim_data['PCA_mean']

initial_condition = sim_data['initial_condition']

try:
    num_timesteps = sim_data['timesteps']
except KeyError:
    num_timesteps = 300

state_dim = x_dim[0][0]

num_simulations = len(state_estimates)

scaling_factor = 1 / np.linalg.norm(M[:,0])

# Calculate NEES and NIS
def calculate_nees_nis(state_estimates, ground_truth, P_prior, P_post, S, y, x_dim, z_dim, true_extent, ssa_func):
    num_simulations = len(state_estimates)
    all_nees = []
    all_nis = []
    all_z_dims = []

    simulations = [6,16,18,54]

    for i in range(num_simulations):
        state_estimates_i = state_estimates[i]
        ground_truth_i = ground_truth[i]
        P_prior_i = P_prior[i]
        P_post_i = P_post[i]
        # S_i = S[i]
        # y_i = y[i]
        x_dim_i = x_dim[i]
        z_dim_i = z_dim[i]

        initial_condition_i = initial_condition[i]

        nees = []
        nis = []

        nees_matrix = []
        nis_matrix = []

        # Add initial condition
        #state_estimates_i.insert(state_estimates_i, 0, initial_condition_i[0], axis=0)
        state_estimates_i.insert(0, [initial_condition_i[0], initial_condition_i[0]]) # added twice to deal with the shape of the array
        ground_truth_i.insert(0, initial_condition_i[2])
        P_post_i.insert(0, initial_condition_i[1])

        for t in range(num_timesteps+1):
            try: 
                state_vector = state_estimates_i[t][-1]
                gt_vector = ground_truth_i[t]

                # if t != 0:
                #     y_it = y_i[t-1]
                #     S_it = S_i[t-1]
                P_it = P_post_i[t]

                # NEES calculation
                estimation_error = state_vector - gt_vector
                estimation_error[2] = ssa_func(estimation_error[2])
                #print(estimation_error[2])
                nees_t = (estimation_error.T @ np.linalg.inv(P_it) @ estimation_error).item()
                nees.append(nees_t)

                # nees_matrix_i = np.diag(estimation_error) @ np.linalg.inv(P) @ np.diag(estimation_error)
                # nees_matrix.append(nees_matrix_i)
                # np.set_printoptions(linewidth=np.inf)
                # print("Covariance matrix: \n", np.array_str(np.log10(P), precision=1))
                # print("NEES matrix: \n", np.array_str(np.log10(nees_matrix_i), precision=1))
                # print("estimation error: \n", np.array_str(estimation_error, precision=1))

                #print("extent covariance: ", np.diag(P[8:,8:]))
                
                # NIS calculation
                # if t != 0:
                #     nis_t = (y_it.T @ np.linalg.inv(S_it) @ y_it).item()
                #     nis.append(nis_t)
                # else:
                #     nis_t = np.nan
                #     nis.append(nis_t)

                # nis_matrix_i = np.diag(y.T) @ np.linalg.inv(S) @ np.diag(y)
                # nis_matrix.append(nis_matrix_i)
                # print("NIS matrix: ", nis_matrix_i)
            except IndexError as e:
                nees_t = np.nan
                nees.append(nees_t)
                nis_t = np.nan
                nis.append(nis_t)  

        if any(abs(n) > 1e20 for n in nees):
            print(f"Sim {i} has NEES values greater than 1e10")
        all_nees.append(nees)
        all_nis.append(nis)
        all_z_dims.append([np.nan, *z_dim_i])

    return all_nees, all_nis, all_z_dims


all_nees, all_nis, all_z_dims = calculate_nees_nis(state_estimates, ground_truth, P_prior, P_post, S, y, x_dim, z_dim, true_extent, ssa)

# Compute ANEES and ANIS
anees = np.nanmean(all_nees, axis=0)
#anis = np.nanmean(all_nis, axis=0)

# Consistency bounds
x_dims = np.array(x_dim[0])
# avg_z_dims = np.nanmean(all_z_dims, axis=0)

nees_upper = chi2.ppf(0.975, state_dim)
nees_lower = chi2.ppf(0.025, state_dim)
# nis_upper = chi2.ppf(0.975, all_z_dims)
# nis_lower = chi2.ppf(0.025, all_z_dims)

anees_upper = gamma.ppf(0.975, a=num_simulations * x_dims / 2, scale=2 / num_simulations)
anees_lower = gamma.ppf(0.025, a=num_simulations * x_dims / 2, scale=2 / num_simulations)

# anis_upper = gamma.ppf(0.975, a=num_simulations * avg_z_dims / 2, scale=2 / num_simulations)
# anis_lower = gamma.ppf(0.025, a=num_simulations * avg_z_dims / 2, scale=2 / num_simulations)

# anis_upper_mean = anis_upper + anis_upper / 2
# anis_lower_mean = anis_lower + anis_lower / 2


# Plot NEES and ANEES comparison
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

time_steps = np.arange(num_timesteps+1) # Convert time steps to seconds

# Column 1: NEES
for i in range(len(all_nees)):
    ax[0].plot(time_steps, all_nees[i], alpha=0.5)
ax[0].plot([], [], color='k', label="Simulations")
ax[0].axhline(nees_upper, color='red', linestyle='--', label="Upper Bound (97,5%)", alpha=0.6, linewidth=3)
ax[0].axhline(nees_lower, color='blue', linestyle='--', label="Lower Bound (2,5%)", alpha=0.6, linewidth=3)
ax[0].set_title("NEES", fontsize=18)
ax[0].set_xlabel("Steps", fontsize=15)
ax[0].set_ylabel("NEES", fontsize=15)
ax[0].tick_params(axis='both', which='major', labelsize=10)
ax[0].grid(True)
ax[0].legend(fontsize=13, loc='upper left')
ax[0].set_xlim([0, time_steps[-1]])
ax[0].set_yscale('log')

# Column 3: ANEES Comparison
ax[1].plot(time_steps, anees, label="ANEES", color="purple")
ax[1].axhline(nees_upper, color='red', linestyle='--', label="Upper Bound (97,5%)", alpha=0.6, linewidth=3)
ax[1].axhline(nees_lower, color='blue', linestyle='--', label="Lower Bound (2,5%)", alpha=0.6, linewidth=3)
ax[1].set_title("ANEES Comparison", fontsize=18)
ax[1].set_xlabel("Steps", fontsize=15)
ax[1].set_ylabel("ANEES", fontsize=15)
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].grid(True)
ax[1].legend(fontsize=13)
ax[1].set_xlim([0, time_steps[-1]])
ax[1].set_yscale('log')

# # Column 1: NEES
# for i in range(len(all_nees)):
#     ax[0, 0].plot(time_steps, all_nees[i], alpha=0.5)
# ax[0, 0].plot([], [], color='k', label="Simulations")
# ax[0, 0].axhline(nees_upper, color='red', linestyle='--', label="Upper Bound (97,5%)", alpha=0.6, linewidth=3)
# ax[0, 0].axhline(nees_lower, color='blue', linestyle='--', label="Lower Bound (2,5%)", alpha=0.6, linewidth=3)
# ax[0, 0].set_title("NEES", fontsize=18)
# ax[0, 0].set_xlabel("Steps", fontsize=15)
# ax[0, 0].set_ylabel("NEES", fontsize=15)
# ax[0, 0].tick_params(axis='both', which='major', labelsize=10)
# ax[0, 0].grid(True)
# ax[0, 0].legend(fontsize=13, loc='upper left')
# ax[0, 0].set_xlim([0, time_steps[-1]])
# ax[0, 0].set_yscale('log')

# # Column 3: ANEES Comparison
# ax[0, 1].plot(time_steps, anees, label="ANEES", color="purple")
# ax[0, 1].axhline(nees_upper, color='red', linestyle='--', label="Upper Bound (97,5%)", alpha=0.6, linewidth=3)
# ax[0, 1].axhline(nees_lower, color='blue', linestyle='--', label="Lower Bound (2,5%)", alpha=0.6, linewidth=3)
# ax[0, 1].set_title("ANEES Comparison", fontsize=18)
# ax[0, 1].set_xlabel("Steps", fontsize=15)
# ax[0, 1].set_ylabel("ANEES", fontsize=15)
# ax[0, 1].tick_params(axis='both', which='major', labelsize=10)
# ax[0, 1].grid(True)
# ax[0, 1].legend(fontsize=13)
# ax[0, 1].set_xlim([0, time_steps[-1]])
# ax[0, 1].set_yscale('log')

# # Column 1: NIS
# for i in range(len(all_nis)):
#     ax[1, 0].plot(time_steps, all_nis[i], alpha=0.5)
#     ax[1, 0].plot(time_steps, nis_upper[i], color='red', linestyle='--', alpha=0.4, linewidth=2)
#     ax[1, 0].plot(time_steps, nis_lower[i], color='blue', linestyle='--', alpha=0.4, linewidth=2)
# ax[1, 0].plot([], [], color='k', label="Simulations")
# ax[1, 0].plot([], [], 'r--', label="Upper Limit (97,5%)")
# ax[1, 0].plot([], [], 'b--', label="Lower Limit (2,5%)")
# ax[1, 0].set_title("EKF NIS", fontsize=18)
# ax[1, 0].set_xlabel("Steps", fontsize=15)
# ax[1, 0].set_ylabel("NIS", fontsize=15)
# ax[1, 0].tick_params(axis='both', which='major', labelsize=10)
# ax[1, 0].grid(True)
# ax[1, 0].legend(fontsize=13)
# ax[1, 0].set_xlim([0, time_steps[-1]])
# ax[1, 0].set_yscale('log')

# # Column 3: ANIS Comparison
# ax[1, 1].plot(time_steps, anis, label="ANIS", color="purple")
# ax[1, 1].plot(time_steps, anis_upper, color='red', linestyle='--', label="Upper Bound (97,5%)", alpha=0.6, linewidth=3)
# ax[1, 1].plot(time_steps, anis_lower, color='blue', linestyle='--', label="Lower Bound (2,5%)", alpha=0.6, linewidth=3)
# ax[1, 1].set_title("ANIS Comparison", fontsize=18)
# ax[1, 1].set_xlabel("Steps", fontsize=15)
# ax[1, 1].set_ylabel("ANIS", fontsize=15)
# ax[1, 1].tick_params(axis='both', which='major', labelsize=10)
# ax[1, 1].grid(True)
# ax[1, 1].legend(fontsize=13)
# ax[1, 1].set_xlim([0, time_steps[-1]])
# ax[1, 1].set_yscale('log')

# Set the title for the entire figure
fig.suptitle(FIGURE_TITLE, fontsize=22)

# Get the max y-limit across both plots (on log scale, use max of actual values, not limits)
ymins = []
ymaxs = []

for a in ax:
    ymin, ymax = a.get_ylim()
    ymins.append(ymin)
    ymaxs.append(ymax)

# Set the same ylim for both plots using the overall min and max
shared_ylim = (min(ymins), max(ymaxs))
for a in ax:
    a.set_ylim(shared_ylim)

# for row in ax:  # Iterate over rows: ax[0] and ax[1]
#     # Get current y-limits for each subplot in the row
#     ymins, ymaxs = [], []
#     for a in row:
#         if a.has_data():  # Skip empty plots
#             ymin, ymax = a.get_ylim()
#             ymins.append(ymin)
#             ymaxs.append(ymax)
    
#     # Set shared y-limits across the row
#     if ymins and ymaxs:
#         shared_ylim = (min(ymins), max(ymaxs))
#         for a in row:
#             if a.has_data():
#                 a.set_ylim(shared_ylim)


# Save and show the figure
plt.tight_layout()
plt.savefig(PLOT_SAVE_PATH, format='svg', dpi=1200)
# plt.show()
