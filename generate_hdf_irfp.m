% module that generates an hdf format file from the analyzed oocyte
% trajectories
% only the x and y values are saved. The keys are the particle indices
% while the values are (2, track_len) sized double dtyped arrays.

%% get list of files
root_dir  = uigetdir('/home/yoonjung/Downloads/Tzer Han/', 'Select the oocytes data folder');
% '/home/yoonjung/Downloads/Tzer Han/oocyte data';
cd(root_dir)
delete *.h5 *.hdf5
file_names_mat = dir(fullfile(root_dir, '*.mat'));
n_files = length(file_names_mat);

%% loop through file to create an hdf file
for i = 1:n_files
    file_name_mat = file_names_mat(i).name;
    [~, file_name, ~] = fileparts(file_name_mat);
    file_name_hdf = [file_name, '.h5'];
    data = load(file_name_mat);
    n_tracks = length(data.posArr);
    for j = 1:n_tracks
        x = data.posArr(j).x';
        y = data.posArr(j).y';
        track_len = length(x);
        h5create(file_name_hdf, ['/', num2str(j)], [track_len, 2], 'Datatype', 'double');
        %h5create(file_name_hdf, ['/', num2str(j)], [track_len, 2], 'Datatype', 'single', ...
        %    'ChunkSize', [...], 'Deflate', 5);
        h5write(file_name_hdf, ['/', num2str(j)], [x, y]);
    end
end
