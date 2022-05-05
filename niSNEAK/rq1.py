from niSneak import sneak_run
from tqdm import tqdm

directories = ['commits/', 'prs/', 'issues/']
files = [['health_bin_data_project0000.csv', 'health_all_data_full_project0000.csv'], ['health_bin_data_project0001.csv', 'health_all_data_full_project0001.csv'], ['health_bin_data_project0002.csv', 'health_all_data_full_project0002.csv'], ['health_bin_data_project0003.csv', 'health_all_data_full_project0003.csv'],
         ['health_bin_data_project0004.csv', 'health_all_data_full_project0004.csv'], ['health_bin_data_project0005.csv', 'health_all_data_full_project0005.csv'], ['health_bin_data_project0006.csv', 'health_all_data_full_project0006.csv'], ['health_bin_data_project0007.csv', 'health_all_data_full_project0007.csv'],
         ['health_bin_data_project0008.csv', 'health_all_data_full_project0008.csv'], ['health_bin_data_project0009.csv', 'health_all_data_full_project0009.csv'], ['health_bin_data_project0010.csv', 'health_all_data_full_project0010.csv'], ['health_bin_data_project0011.csv', 'health_all_data_full_project0011.csv']]
pbar1 = tqdm(directories)
pbar2 = tqdm(files, position=0)
for directory in pbar1:
    for file in pbar2:
        pbar2.set_description("Processing: " + file[0][-15:-4])
        sneak_run([file[0]], [file[1]], directory, True)