import scat_utils as scu

root_dir = './data/'
file_name = 'tbd_2.pt'
avg_len = 2**8
log_transform = True
n_filter_octave = [1, 1]

scu.scat_transform(file_name, avg_len=avg_len, log_transform=log_transform,
    n_filter_octave=n_filter_octave, save_file=True, root_dir=root_dir)
