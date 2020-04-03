'''module that simulates the two beads data and trains the LSTM and scat-transform + LSTM model'''
# parameters that represent the conditions are sampled from a continuous space
import os
import scat_utils as scu
import sim_utils as siu

'''custom libraries'''
import common_utils as cu
ROOT_DIR = './data/simulations/data_len_2048_gamma_1_3_k_1_7_t_4_10'

# common inputs
data_len = 2**11
avg_lens = [2**8]
n_filter_octaves = [(1,1)]
dt = 0.001
n_datas = [20, 50, 200, 500]
n_data_test = 1000

root_dir = ROOT_DIR

gamma_low = 1
gamma_high = 3

k_ratio_low = 1
k_ratio_high = 7 # 4 or 7

diff_coef_ratio_low = 4
diff_coef_ratio_high = 10 # 7 or 10

# simulate two beads
for n_data in n_datas:
    print("simulating training data for n_data:{}".format(n_data))
    file_name_data = siu.sim_two_beads_sample(data_len,
            gammas=[gamma_low, gamma_high],
            k_ratios=[k_ratio_low, k_ratio_high],
            diff_coef_ratios=[diff_coef_ratio_low, diff_coef_ratio_high],
            dt=dt, n_data=n_data, n_steps_initial=10000,
            save_file=True, root_dir=root_dir, dtype='float32')
    for avg_len in avg_lens:
        for n_filter_octave in n_filter_octaves:
            print("\tscat transforming with parameters avg_len:{}, n_filter_octave:{}".format(avg_len, n_filter_octave))
            file_name_scat = scu.scat_transform(file_name_data, avg_len, log_transform=False, n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)

# simulate data for testing performance
print("simulating test data for n_data_test:{}".format(n_data_test))
file_name_data_test_old = siu.sim_two_beads_sample(data_len,
        gammas=[gamma_low, gamma_high],
        k_ratios=[k_ratio_low, k_ratio_high],
        diff_coef_ratios=[diff_coef_ratio_low, diff_coef_ratio_high],
        dt=dt, n_data=n_data_test, n_steps_initial=10000,
        save_file=True, root_dir=root_dir, dtype='float32')

nums = cu.match_filename(r'tbd_([0-9]+)_test.pt', root_dir=root_dir)
nums = [int(num) for num in nums]
idx = max(nums) + 1 if nums else 0

file_name_data_test_new = 'tbd_{}_test.pt'.format(idx)
os.rename(os.path.join(root_dir, file_name_data_test_old), os.path.join(root_dir, file_name_data_test_new))

# scat transforming test data
for avg_len in avg_lens:
    for n_filter_octave in n_filter_octaves:
        print("\tscat transforming with parameters avg_len:{}, n_filter_octave:{}".format(avg_len, n_filter_octave))
        file_name_scat_test = scu.scat_transform(file_name_data_test_new,
                avg_len, log_transform=False, n_filter_octave=n_filter_octave,
                save_file=True, root_dir=root_dir)
