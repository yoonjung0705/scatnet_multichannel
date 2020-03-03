'''finds the jobs that exited with failure and appends them to params.csv as new jobs'''
'''standard imports'''
import os
import torch
import re
import glob

ROOT_DIR='/nobackup/users/yoonjung/repos/scatnet_multichannel/data/simulations'
file_name_params = 'params.csv'
file_name_regex = 'tbd_*_meta_rnn_*_k_ratios.pt'
file_name_data_regex = r'(tbd_[0-9]+)_meta_rnn_.*_k_ratios.pt'
root_dir=ROOT_DIR
epochs_len = 200 # threshold for determining whether the training failed or not
# epochs_len does not mean the 'n_epochs_max' value. 'epochs' is a list of the epoch number
# and therefore if you log the results every 10 epochs, it'll be a list with length 200 if n_epochs_max is 2000
label_names = ['k_ratios', 'diff_coef_ratios']
file_count = 0

'''fields of the params.csv file'''
#jobid
#submit_count
#file_name
#root_dir
#hidden_size
#n_layers
#bidirectional
#classifier
#idx_label = 
#epochs = 2000
#train_ratio = 0.8
batch_size_exit = 32
#n_workers = 4
lr = 0.001
betas = "0.9 0.999"
opt_level = "O2"
seed = 42
log_interval = 10

file_paths = glob.glob(os.path.join(root_dir, file_name_regex))
file_paths_exit = []
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    match = re.fullmatch(file_name_data_regex, file_name)
    if match is None: raise IOError("Invalid regular expression give: no match found for data file name")
    file_name_data = match.group(1) + '.pt'
    meta = torch.load(file_path)
    bidirectional = "--bidirectional" if meta['bidirectional'] else ""
    classifier = "--classifier" if meta['classifier'] else ""
    idx_label = 0 if label_names[0] in file_name else 1
    if len(meta['epoch']) < epochs_len:
        row = "\n,1,{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}".format(
                file_name_data,
                root_dir,
                meta['hidden_size'],
                meta['n_layers'],
                bidirectional,
                classifier,
                idx_label,
                meta['n_epochs_max'],
                meta['train_ratio'],
                batch_size_exit,
                meta['n_workers'],
                lr,
                betas,
                opt_level,
                seed,
                log_interval)

        with open(file_name_params, 'a') as f:
            f.write(row)
        file_count += 1

print("{} training jobs appended to params.csv".format(file_count))

        
        


