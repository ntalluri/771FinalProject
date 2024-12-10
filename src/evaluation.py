import pandas as pd 
# take the trained encoder, add a binary classifier nn on top to classify 1 for event 0 for no event

def get_all_file_paths(folder_paths):
    all_file_paths = []
    for subdir_path in folder_paths:
        # list files ending with '.h5' in the current subdirectory
        try:
            files_in_subdir = os.listdir(subdir_path)
        except FileNotFoundError:
            print(f"Subdirectory not found: {subdir_path}")
            continue

        h5_files = [
            os.path.join(subdir_path, f)
            for f in files_in_subdir
            if os.path.isfile(os.path.join(subdir_path, f)) and f.endswith('.h5')
        ]
        all_file_paths.extend(h5_files)
    return all_file_paths

def load_labels(csv_path):
    """
    Load labels from CSV file
    """
    labels_df = pd.read_csv('labeled_filenames.csv')
    return dict(zip(labels_df['Filename'], labels_df['Label']))

# run code

# have a labeled dataset in labeled_filenames.csv
# the subfolders have the folders that contain each file

# grab the labels dict
labels_dict = load_labels("labeled_filenames.csv")
print(labels_dict)

# grab the subfolders
folder_paths = []
with open('labeled_folders.txt', 'r') as file:
    folder_paths = file.read().splitlines()

# get all the H5 files 
all_file_paths = get_all_file_paths(folder_paths)

# grab all files, and create x, y pairs? (x is the input, y is the label) using the labeled_filenames.csv
