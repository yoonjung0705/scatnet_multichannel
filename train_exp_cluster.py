'''module that processes and trains a classifier for the optical trap active bath experiment data'''
# TODO: inputs: file_name_data, hidden_size, n_layers, bidirectonal, n_epochs_max, train_ratio, batch_size, n_workers, lr, betas
# use default ROOT_DIR as root_dir
# depending on the given file_name_data, do either raw data training or scat data training
import net_utils as nu

print("training rnn for {}, avg_len:{}, n_filter_octave:{}, hidden_size:{}, n_layers:{}, bidirectional:{}"
    .format(file_name_scat, avg_len, n_filter_octave, hidden_size, n_layers, bidirectional))
nu.train_rnn(file_name_scat, hidden_size, n_layers, bidirectional, classifier=True,
    n_epochs_max=n_epochs_max, train_ratio=train_ratio, batch_size=batch_size,
    n_workers=n_workers, root_dir=root_dir, lr=lr, betas=betas)

#print("training rnn for {}, hidden_size:{}, n_layers:{}, bidirectional:{}".format(file_name_data, hidden_size, n_layers, bidirectional))
#nu.train_rnn(file_name_data, hidden_size, n_layers, bidirectional, classifier=True,
#    n_epochs_max=n_epochs_max, train_ratio=train_ratio, batch_size=batch_size,
#    n_workers=n_workers, root_dir=root_dir, lr=lr, betas=betas)


