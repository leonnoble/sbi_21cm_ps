import os
import pickle

import get_ps_21cmEMU
import numpy as np
from getdist import MCSamples, plots
import torch
_ = torch.manual_seed(0)
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

METHOD = "nle"

# calling the emulator and getting the 1D power spectrum training data set

redshift_value = 8.01
# fiducial parameters for the emulator
log_f_star10 = -1.2
alpha_star = 0.5
t_star = 0.55
log_f_esc10 = -1.3
alpha_esc = 0
log_f_star7_mini = -2.5
log_f_esc7_mini = -1.5
log_l_x = 40.0
log_l_x_mini = 41.5
nu_x_thresh = 500
sigma_8 = 0.8118


astro_params = {"F_STAR10": log_f_star10,
          'ALPHA_STAR': alpha_star,
          't_STAR': t_star,
          'F_ESC10': log_f_esc10,
          'ALPHA_ESC': alpha_esc,
          'F_STAR7_MINI': log_f_star7_mini,
          'F_ESC7_MINI': log_f_esc7_mini,
          'L_X': log_l_x,
          'L_X_MINI': log_l_x_mini,
          'NU_X_THRESH': nu_x_thresh,
          'SIGMA_8': sigma_8
         }

noise_sigma = np.load("data/sigma_pk_z8.01.npy")


def get_traning_data(theta, num_simulations):
    "first varying only three parameters: F_STAR10, F_ESC10, ALPHA_STAR"
    if torch.is_tensor(theta):
        theta = theta.detach().cpu().numpy()
    f_star_10 = theta[:, 0]
    f_esc_10 = theta[:, 1]
    alpha_star = theta[:, 2]

    ps_data = np.zeros((num_simulations, 32))

    for i in range(num_simulations):
        astro_params = {"F_STAR10": f_star_10[i],
          'ALPHA_STAR': alpha_star[i],
          't_STAR': t_star,
          'F_ESC10': f_esc_10[i],
          'ALPHA_ESC': alpha_esc,
          'F_STAR7_MINI': log_f_star7_mini,
          'F_ESC7_MINI': log_f_esc7_mini,
          'L_X': log_l_x,
          'L_X_MINI': log_l_x_mini,
          'NU_X_THRESH': nu_x_thresh,
          'SIGMA_8': sigma_8
         }
        ps = get_ps_21cmEMU.simu_1d_ps(astro_params, redshift_value=redshift_value)

        np.random.seed(i)  # for reproducibility
        ps_noisy = ps + np.random.normal(0, noise_sigma, size=ps.shape)  # add noise to the power spectrum
        ps_data[i, :] = ps_noisy

    return ps_data


# defining the parameter space for training data generation
from sbi.utils import BoxUniform
lower_bounds = np.array([-2.0, -3.0, 0.0])  # lower bounds for F_STAR10, F_ESC10, ALPHA_STAR
upper_bounds = np.array([-0.5, 0.0, 1.0])  # upper bounds for F_STAR10, F_ESC10, ALPHA_STAR
prior = BoxUniform(low=torch.tensor(lower_bounds), high=torch.tensor(upper_bounds), device=device)

# choosing the sbi method for inference: NLE
from sbi.inference import NLE
inference = NLE(prior=prior, device=device)

# loading the training data set
theta = np.load('data/theta_training_data_f_star10_f_esc10_alpha_star_ska_1080_num_simulations_30000.npy')
x = np.load('data/ps_training_data_f_star10_f_esc10_alpha_star_ska_1080_num_simulations_30000.npy')
x_torch = torch.tensor(x, dtype=torch.float32, device=device)
theta_torch = torch.tensor(theta, dtype=torch.float32, device=device)

# training the density estimator with the training data set
inference = inference.append_simulations(theta_torch, x_torch)
density_estimator = inference.train()
print("Training complete.")

posterior = inference.build_posterior()
print(posterior)

with open(f"data/{METHOD}_posterior.pkl", "wb") as handle:
    pickle.dump(posterior, handle)
print(f"Saved trained posterior to data/{METHOD}_posterior.pkl")

ps_obs = get_ps_21cmEMU.simu_1d_ps(astro_params, redshift_value=redshift_value)  # test data for inference
ps_obs_torch = torch.tensor(ps_obs, dtype=torch.float32, device=device)

posterior_samples = posterior.sample((1000,), x=ps_obs_torch)  # draw 1000 posterior samples

labels = [r'\log_{10} F_{*,10}', r'\log_{10} F_{\rm esc,10}', r'\alpha_{*}']
names = ['F_STAR10', 'F_ESC10', 'ALPHA_STAR']
true_params = [log_f_star10, log_f_esc10, alpha_star]

samples = MCSamples(
    samples=posterior_samples.cpu().numpy(),
    names=names,
    labels=labels
)

# dictionary mapping the parameter names to their true values
markers_dict = {name: val for name, val in zip(names, true_params)}

g = plots.get_subplot_plotter()
g.triangle_plot(samples, filled=True, markers=markers_dict)
g.export(f"plots/{METHOD}_triangle.png")
print(f"Saved plots/{METHOD}_triangle.png")

# Posterior Predictive Checks

theta_posterior_ppc = posterior.sample((200,), x=ps_obs_torch)  # draw 200 posterior samples for posterior predictive check
x_predictive = get_traning_data(theta_posterior_ppc, 200)  # generate predictive data from the posterior samples
x_predictive = torch.tensor(x_predictive, dtype=torch.float32)

print("Posterior predictives: ", torch.mean(x_predictive, axis=0))
print("Observation: ", ps_obs_torch)

k_values = np.load('data/k_values.npy')
plt.figure()
plt.plot(k_values, torch.mean(x_predictive, axis=0), label='Posterior Predictive Mean', color='green')
plt.plot(k_values, ps_obs_torch.cpu(), label='Observation', color='orange', alpha=0.7)
plt.xscale('log')
plt.xlabel(r'$k$ [Mpc$^{-1}$]')
plt.ylabel(r'$\Delta^2(k)$ [mK$^2$]')
plt.legend()
plt.savefig(f"plots/{METHOD}_posterior_predictive.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved plots/{METHOD}_posterior_predictive.png")

# Simulation-Based Calibration (SBC)

from sbi.diagnostics import run_sbc
from sbi.analysis.plot import sbc_rank_plot

num_sbc_samples = 200  # number of sbc runs
prior_samples = prior.sample((num_sbc_samples,))
prior_predictives = get_traning_data(prior_samples, num_sbc_samples)  # generate predictive data from the prior samples

num_posterior_samples = 1_000
ranks, dap_samples = run_sbc(
    prior_samples,
    torch.tensor(prior_predictives, dtype=torch.float32, device=device),
    posterior,
    num_posterior_samples=num_posterior_samples,
    use_batched_sampling=False,  # `True` can give speed-ups, but can cause memory issues.
)

np.savez(
    f"data/{METHOD}_sbc_results.npz",
    ranks=ranks.cpu().numpy(),
    dap_samples=dap_samples.cpu().numpy(),
    num_posterior_samples=num_posterior_samples,
)
print(f"Saved data/{METHOD}_sbc_results.npz")

fig, ax = sbc_rank_plot(
    ranks=ranks.cpu(),
    num_posterior_samples=num_posterior_samples,
    plot_type="hist",
    num_bins=None,
)
fig.savefig(f"plots/{METHOD}_sbc_rank_hist.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved plots/{METHOD}_sbc_rank_hist.png")

fig, ax = sbc_rank_plot(
    ranks,
    num_posterior_samples,
    num_bins=20,
    figsize=(10, 10),
)
fig.savefig(f"plots/{METHOD}_sbc_rank_20bins.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved plots/{METHOD}_sbc_rank_20bins.png")

# Posterior calibration with TARP

from sbi.diagnostics import run_tarp
from sbi.analysis.plot import plot_tarp

num_tarp_samples = 200
thetas = prior.sample((num_tarp_samples,))
xs = get_traning_data(thetas, num_tarp_samples)

ecp, alpha = run_tarp(
    thetas,
    torch.tensor(xs, dtype=torch.float32, device=device),
    posterior,
    references=None,  # will be calculated automatically.
    num_posterior_samples=1000,
)

np.savez(
    f"data/{METHOD}_tarp_results.npz",
    ecp=ecp.cpu().numpy(),
    alpha=alpha.cpu().numpy(),
)
print(f"Saved data/{METHOD}_tarp_results.npz")

fig, ax = plot_tarp(ecp.cpu(), alpha.cpu())
fig.savefig(f"plots/{METHOD}_tarp.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved plots/{METHOD}_tarp.png")
