import numpy as np
import sim_utils as siu

ROOT_DIR = './data/'

# inputs common to different simulations
data_len = 2**11
dt = 0.01 # REVIEW
n_data = 100

# simulate brownian
diff_coefs = np.arange(4,8, 0.5)
siu.sim_brownian(data_len, diff_coefs, dt, n_data, save_file=True, root_dir=ROOT_DIR)

# simulate one bead
diff_coefs = np.arange(4,8, 0.5)
ks = np.arange(1,3,0.5)
siu.sim_one_bead(data_len, diff_coefs, ks, dt, n_data, n_steps_initial=10000, save_file=True, root_dir=ROOT_DIR)

# simulate poisson
lams = np.arange(4,8, 0.5)
siu.sim_poisson(data_len, lams, dt, n_data, save_file=True, root_dir=ROOT_DIR)

# simulate two beads
k_ratios = np.arange(1,3,0.5)
diff_coef_ratios = np.arange(2,8,1)
siu.sim_two_beads(data_len, k_ratios, diff_coef_ratios, dt, n_data, n_steps_initial=10000, save_file=True, root_dir=ROOT_DIR)

