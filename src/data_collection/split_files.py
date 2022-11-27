import os
import numpy as np

def split(train_size, valid_size, test_size):
    # puts all the images in the correct folders
    assert train_size + valid_size + test_size == 1

    cwd = os.getcwd()
    
    # get all the files
    files = os.listdir(cwd)
    files = [f for f in files if f.endswith(".png")]

    # shuffle the files
    np.random.shuffle(files)

    # split the files
    train_files = files[:int(train_size * len(files))]
    valid_files = files[int(train_size * len(files)):int((train_size + valid_size) * len(files))]
    test_files = files[int((train_size + valid_size) * len(files)):]

    # move the files
    for f in train_files:
        os.rename(f, "train/" + f)
    for f in valid_files:    
        os.rename(f, "valid/" + f)
    for f in test_files:
        os.rename(f, "test/" + f)

if __name__ == "__main__":
    split(0.8, 0.1, 0.1)