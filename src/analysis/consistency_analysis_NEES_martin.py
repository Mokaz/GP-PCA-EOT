import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, gamma

import numpy.linalg as npl
from scipy.linalg import cholesky, solve_triangular

# -------------------------------
# Helpers for stable NEES math
# -------------------------------

def symmetrize(P):
    return 0.5 * (P + P.T)

def cholesky_solve(P, e, jitter=1e-9, max_tries=5):
    """
    Solve P^{-1} e via Cholesky without explicitly inverting P.
    Returns: Pinv_e, y (whitened = L^{-1} e), L (Cholesky), ok_flag
    """
    P = symmetrize(P)
    tries = 0
    while tries <= max_tries:
        try:
            L = cholesky(P, lower=True, check_finite=False)
            y = solve_triangular(L, e, lower=True, check_finite=False)
            Pinv_e = solve_triangular(L.T, y, lower=False, check_finite=False)
            return Pinv_e, y, L, True
        except Exception:
            # Add diagonal jitter if SPD is borderline
            P = P + (jitter * (10**tries)) * np.eye(P.shape[0])
            tries += 1
    return None, None, None, False

def nees_decompose_per_timestep(e, P):
    """
    Returns: nees, c (per-state additive contributions), y2 (whitened squares),
             lam_min, condP, ok_flag
    """
    Pinv_e, y, L, ok = cholesky_solve(P, e)
    if not ok:
        d = e.shape[0]
        return (np.nan,
                np.full(d, np.nan),
                np.full(d, np.nan),
                np.nan,
                np.nan,
                False)

    # Additive contributions in native coordinates (sum to NEES)
    c = e * Pinv_e
    y2 = y**2
    nees = float(e @ Pinv_e)

    # Conditioning diagnostics: eigenvalues via SVD of L (since P = L L^T)
    try:
        S = npl.svd(L, compute_uv=False)   # singular values of L
        lam_min = (S.min())**2
        lam_max = (S.max())**2
        condP = lam_max / lam_min if lam_min > 0 else np.inf
    except Exception:
        lam_min, condP = np.nan, np.nan

    return nees, c, y2, lam_min, condP, True

# -------------------------------
# Project + data load
# -------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

NAME = 'bfgs'
# FILENAME = f'simulation_data_{NAME}.pkl'
FILENAME = "simulation_data_martin_sim_bfgs_ellipse_90frames.pkl"
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

PLOT_SAVE_PATH = os.path.join(FIG_DIR, f'consistency_{FILENAME}.svg')
FIGURE_TITLE = f'Consistency Analysis: {FILENAME}'

from utils.tools import ssa  # your angle wrap function

# Load the simulation data (expects a dict-like npz with pickled arrays/lists)
sim_data = np.load(f'data/results/{FILENAME}', allow_pickle=True)

# Extract the data
state_estimates = sim_data['state_estimates']
ground_truth    = sim_data['ground_truth']
static_covariances = sim_data['static_covariances']
true_extent     = sim_data['true_extent']
P_prior         = sim_data['P_prior']
P_post          = sim_data['P_post']
S               = sim_data['S']
y               = sim_data['y']
z               = sim_data['z']
x_dim           = sim_data['x_dim']
z_dim           = sim_data['z_dim']

M                  = sim_data['PCA_eigenvectors']
fourier_coeff_mean = sim_data['PCA_mean']
initial_condition  = sim_data['initial_condition']

print(type(state_estimates[0]))
print(len(state_estimates[0]))
print(state_estimates[0])

exit()

try:
    num_timesteps = int(sim_data['timesteps'])
except KeyError:
    num_timesteps = 300

# state_dim might be wrapped in arrays; coerce to int
state_dim = int(np.array(x_dim[0][0]).item() if np.ndim(x_dim[0]) > 0 else x_dim[0])

num_simulations = len(state_estimates)
_ = 1 / np.linalg.norm(M[:, 0])  # (scaling_factor, not used below)

# -------------------------------
# NEES/NIS + per-state decomposition
# -------------------------------

def calculate_nees_nis(state_estimates, ground_truth, P_prior, P_post, S, y,
                       x_dim, z_dim, true_extent, ssa_func, num_timesteps, state_dim):
    num_simulations = len(state_estimates)

    all_nees = []     # list of length N; each entry: (T,) NEES over time
    all_nis  = []     # (not used; placeholder)
    all_z_dims = []   # (not used; placeholder)

    # Per-state time series (for every sim): arrays shaped (T x d)
    all_c_series   = []   # e_i * (P^{-1} e)_i
    all_y2_series  = []   # whitened components squared (y_i^2)
    all_lammin     = []   # λ_min(P) per time
    all_cond       = []   # cond(P) per time

    for i in range(num_simulations):
        # Ensure lists for mutation
        state_estimates_i = list(state_estimates[i])
        ground_truth_i    = list(ground_truth[i])
        P_post_i          = list(P_post[i])

        # Insert initial condition at t=0, matching your original approach
        ic = initial_condition[i]
        state_estimates_i.insert(0, [ic[0], ic[0]])  # added twice to match original shape convention
        ground_truth_i.insert(0, ic[2])
        P_post_i.insert(0, ic[1])

        T = min(len(state_estimates_i), num_timesteps + 1)
        d = state_dim

        nees = np.full(T, np.nan)
        c_series   = np.full((T, d), np.nan)
        y2_series  = np.full((T, d), np.nan)
        lammin_ser = np.full(T, np.nan)
        cond_ser   = np.full(T, np.nan)

        for t in range(T):
            try:
                xhat = np.array(state_estimates_i[t][-1], dtype=float)
                xgt  = np.array(ground_truth_i[t], dtype=float)
                P    = np.array(P_post_i[t], dtype=float)

                e = (xhat - xgt)
                # wrap heading (index 2) with ssa()
                if d >= 3:
                    e[2] = ssa_func(e[2])

                nees_t, c_t, y2_t, lam_min, condP, ok = nees_decompose_per_timestep(e, P)
                nees[t]       = nees_t
                c_series[t,:] = c_t
                y2_series[t,:]= y2_t
                lammin_ser[t] = lam_min
                cond_ser[t]   = condP
            except Exception:
                # leave NaNs for this timestep
                pass

        if np.any(np.abs(nees) > 1e20):
            print(f"[Warn] Simulation {i}: NEES has values > 1e20")

        all_nees.append(nees)
        all_nis.append(np.full(T, np.nan))  # if you want NIS later, fill it here
        all_z_dims.append([np.nan, *z_dim[i]])

        all_c_series.append(c_series)
        all_y2_series.append(y2_series)
        all_lammin.append(lammin_ser)
        all_cond.append(cond_ser)

    return all_nees, all_nis, all_z_dims, all_c_series, all_y2_series, all_lammin, all_cond


all_nees, all_nis, all_z_dims, all_c_series, all_y2_series, all_lammin, all_cond = \
    calculate_nees_nis(state_estimates, ground_truth, P_prior, P_post, S, y,
                       x_dim, z_dim, true_extent, ssa, num_timesteps, state_dim)

# -------------------------------
# Determinant plots (avg across sims)
# -------------------------------

def plot_covariance_determinants(P_prior, P_post, name):
    """
    Calculates and plots the average determinant of prior/posterior covariance
    across simulations (per timestep) on a log scale.
    """
    P_prior = np.array(P_prior, dtype=object)
    P_post  = np.array(P_post,  dtype=object)

    # det over each sim/time — handle object dtype (lists of arrays)
    det_prior_all_sims = []
    det_post_all_sims  = []

    for Ps in P_prior:
        det_prior_all_sims.append(np.array([np.linalg.det(np.array(P, dtype=float)) for P in Ps]))
    for Ps in P_post:
        det_post_all_sims.append(np.array([np.linalg.det(np.array(P, dtype=float)) for P in Ps]))

    # Pad to same length if needed (use num_timesteps)
    det_prior_all_sims = [d[:num_timesteps] for d in det_prior_all_sims]
    det_post_all_sims  = [d[:num_timesteps] for d in det_post_all_sims]

    det_prior_all_sims = np.vstack(det_prior_all_sims)  # N x T
    det_post_all_sims  = np.vstack(det_post_all_sims)

    avg_det_prior = np.nanmean(det_prior_all_sims, axis=0)
    avg_det_post  = np.nanmean(det_post_all_sims,  axis=0)

    time_axis = np.arange(len(avg_det_prior))

    plt.figure(figsize=(12, 8))
    plt.plot(time_axis, avg_det_prior, label='Avg. det(P_prior)')
    plt.plot(time_axis, avg_det_post,  label='Avg. det(P_post)')
    plt.xlabel("Timesteps", fontsize=14)
    plt.ylabel("Determinant (log scale)", fontsize=14)
    plt.title(f"Evolution of Covariance Determinants — {name}", fontsize=16)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, f'determinant_plot_{name}.svg')
    plt.savefig(save_path, format='svg', dpi=1200)
    print(f"Covariance determinant plot saved to {save_path}")
    plt.close()

plot_covariance_determinants(P_prior, P_post, NAME)

# -------------------------------
# NEES & ANEES plots (global)
# -------------------------------

anees = np.nanmean(np.stack(all_nees, axis=0), axis=0)  # average across simulations

# Consistency bounds (single-run NEES bands)
nees_upper = chi2.ppf(0.975, df=state_dim)
nees_lower = chi2.ppf(0.025, df=state_dim)

# ANEES 95% bands for average over N sims:
# Sum of NEES over sims ~ ChiSq(N*state_dim); Average has Gamma(shape=N*state_dim/2, scale=2/N)
anees_upper = gamma.ppf(0.975, a=(num_simulations * state_dim) / 2.0, scale=2.0 / num_simulations)
anees_lower = gamma.ppf(0.025, a=(num_simulations * state_dim) / 2.0, scale=2.0 / num_simulations)

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
time_steps = np.arange(num_timesteps + 1)

# NEES per simulation
for i in range(len(all_nees)):
    series = all_nees[i]
    T = min(len(series), len(time_steps))
    ax[0].plot(time_steps[:T], series[:T], alpha=0.5)
    # This part is just for demonstration if you want to save each series
    # to its own file, which might create many files.
    # A better approach is to save all series into one file after the loop.
    os.makedirs(os.path.join(FIG_DIR, 'nees_csvs'), exist_ok=True)
    series_to_save = np.vstack((time_steps[:T], series[:T])).T
    np.savetxt(os.path.join(FIG_DIR, 'nees_csvs', f'nees_sim_{i}.csv'), series_to_save, delimiter=',', header='timestep,nees', comments='')
ax[0].plot([], [], color='k', label="Simulations")
ax[0].axhline(nees_upper, color='red', linestyle='--', label="NEES Upper (97.5%)", alpha=0.7, linewidth=2.5)
ax[0].axhline(nees_lower, color='blue', linestyle='--', label="NEES Lower (2.5%)",  alpha=0.7, linewidth=2.5)
ax[0].set_title("NEES", fontsize=18)
ax[0].set_xlabel("Steps", fontsize=15)
ax[0].set_ylabel("NEES", fontsize=15)
ax[0].grid(True, which="both", ls="--")
ax[0].legend(fontsize=12, loc='upper left')
ax[0].set_xlim([0, time_steps[-1]])
ax[0].set_yscale('log')

# ANEES
T = min(len(anees), len(time_steps))
ax[1].plot(time_steps[:T], anees[:T], label="ANEES", color="purple")
ax[1].axhline(anees_upper, color='red', linestyle='--', label="ANEES Upper (97.5%)", alpha=0.7, linewidth=2.5)
ax[1].axhline(anees_lower, color='blue', linestyle='--', label="ANEES Lower (2.5%)",  alpha=0.7, linewidth=2.5)
ax[1].set_title("ANEES Comparison", fontsize=18)
ax[1].set_xlabel("Steps", fontsize=15)
ax[1].set_ylabel("ANEES", fontsize=15)
ax[1].grid(True, which="both", ls="--")
ax[1].legend(fontsize=12)
ax[1].set_xlim([0, time_steps[-1]])
ax[1].set_yscale('log')

fig.suptitle(FIGURE_TITLE, fontsize=22)

# Normalize y-lims across the two subplots (after plotting)
ymins, ymaxs = [], []
for a in ax:
    ymin, ymax = a.get_ylim()
    ymins.append(ymin); ymaxs.append(ymax)
shared_ylim = (min(ymins), max(ymaxs))
for a in ax:
    a.set_ylim(shared_ylim)

plt.tight_layout()
plt.savefig(PLOT_SAVE_PATH, format='svg', dpi=1200)
print(f"NEES/ANEES plots saved to {PLOT_SAVE_PATH}")
plt.close()

# -------------------------------
# Per-state plots & conditioning
# -------------------------------

def summarize_across_sims(arr_list):
    """
    Stack list of (T x d) -> (N x T x d) and return median & IQR over sims.
    """
    A = np.stack(arr_list, axis=0)  # N x T x d
    med = np.nanmedian(A, axis=0)
    q25 = np.nanpercentile(A, 25, axis=0)
    q75 = np.nanpercentile(A, 75, axis=0)
    return med, q25, q75

# Provide labels (override with your actual names if you want)
state_labels = [
    "x0 (North pos)", "x1 (East pos)", "x2 (Heading)",
    "x3 (North vel)", "x4 (East vel)", "x5 (Yaw rate)",
    "x6 (Length)", "x7 (Width)",
    "x8 (PCA 1)", "x9 (PCA 2)", "x10 (PCA 3)", "x11 (PCA 4)"
]
# Fallback if state dimension is not 12
if len(state_labels) != state_dim:
    print(f"[Warning] Pre-defined labels length ({len(state_labels)}) does not match state_dim ({state_dim}). Using generic labels.")
    state_labels = [f"x{i}" for i in range(int(state_dim))]

def plot_per_state_series(time, series_list, title, ylabel, state_labels,
                          ci95=None, overlay_sims=False, logy=False, outfile=None):
    """
    series_list: list over sims of (T x d)
    """
    med, q25, q75 = summarize_across_sims(series_list)
    T, d = med.shape

    cols = min(4, d)
    rows = int(np.ceil(d / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols+1, 2.8*rows+1), squeeze=False)
    axes = axes.flatten()

    for i in range(d):
        ax = axes[i]
        if overlay_sims:
            for S in series_list:
                Ti = min(S.shape[0], len(time))
                ax.plot(time[:Ti], S[:Ti, i], alpha=0.25, linewidth=0.8)
        Ti = min(T, len(time))
        ax.plot(time[:Ti], med[:Ti, i], label="median", linewidth=2)
        ax.fill_between(time[:Ti], q25[:Ti, i], q75[:Ti, i], alpha=0.2, label="IQR")
        if ci95 is not None:
            ax.axhline(ci95, ls="--", color="r", alpha=0.7, label="95% χ²₁" if i == 0 else None)
        ax.set_title(state_labels[i])
        ax.set_xlabel("Steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        if logy:
            ax.set_yscale("log")
        if i == 0:
            ax.legend()

    # Hide unused subplots
    for j in range(d, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if outfile:
        plt.savefig(outfile, format='svg', dpi=1200)
        print(f"Saved: {outfile}")
    plt.close()

time_steps = np.arange(num_timesteps + 1)

# Per-state additive contributions (sum to NEES)
plot_per_state_series(
    time=time_steps,
    series_list=all_c_series,
    title=f"Per-State NEES Contributions — {NAME}",
    ylabel="c_i = e_i · (P^{-1}e)_i",
    state_labels=state_labels,
    ci95=None,                       # no chi-square line for c_i
    overlay_sims=False,
    logy=False,
    outfile=os.path.join(FIG_DIR, f"consistency_{FILENAME}_per_state_contribs.svg")
)

# Whitened per-state squares (each ~ χ²₁, so 95% at 3.841)
plot_per_state_series(
    time=time_steps,
    series_list=all_y2_series,
    title=f"Whitened Per-State Squares — {NAME}",
    ylabel="y_i^2",
    state_labels=state_labels,
    ci95=chi2.ppf(0.95, df=1),       # 3.841
    overlay_sims=False,
    logy=True,                       # spikes easier to see on log-scale
    outfile=os.path.join(FIG_DIR, f"consistency_{FILENAME}_per_state_whitened.svg")
)

def plot_conditioning(time, lammin_list, cond_list, name):
    med_lam, q25_lam, q75_lam = summarize_across_sims(lammin_list)  # (T,)
    med_cond, q25_cond, q75_cond = summarize_across_sims(cond_list) # (T,)

    T = min(len(med_lam), len(time))
    U = min(len(med_cond), len(time))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(time[:T], med_lam[:T], label="λ_min median")
    ax[0].fill_between(time[:T], q25_lam[:T], q75_lam[:T], alpha=0.2, label="IQR")
    ax[0].set_title("Min eigenvalue of P")
    ax[0].set_xlabel("Steps"); ax[0].set_ylabel("λ_min(P)")
    ax[0].set_yscale("log"); ax[0].grid(True, which="both", ls="--"); ax[0].legend()

    ax[1].plot(time[:U], med_cond[:U], label="cond(P) median")
    ax[1].fill_between(time[:U], q25_cond[:U], q75_cond[:U], alpha=0.2, label="IQR")
    ax[1].set_title("Condition number of P")
    ax[1].set_xlabel("Steps"); ax[1].set_ylabel("cond(P)")
    ax[1].set_yscale("log"); ax[1].grid(True, which="both", ls="--"); ax[1].legend()

    fig.suptitle(f"Covariance Conditioning — {name}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = os.path.join(FIG_DIR, f"conditioning_{name}.svg")
    plt.savefig(out, format='svg', dpi=1200)
    print(f"Saved: {out}")
    plt.close()

# plot_conditioning(time_steps, all_lammin, all_cond, NAME)
