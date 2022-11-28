import os
import numpy as np

def split(train_size, valid_size, test_size, path):
    # puts all the images in the correct folders
    assert train_size + valid_size + test_size == 1
    
    # get all the files
    files = os.listdir(path)
    files = [f for f in files if f.endswith(".png")]

    # shuffle the files
    np.random.shuffle(files)

    # split the files
    train_files = files[:int(train_size * len(files))]
    valid_files = files[int(train_size * len(files)):int((train_size + valid_size) * len(files))]
    test_files = files[int((train_size + valid_size) * len(files)):]

    # make the directories
    os.makedirs(os.path.join(path, "train"), exist_ok=True)
    os.makedirs(os.path.join(path, "valid"), exist_ok=True)
    os.makedirs(os.path.join(path, "test"), exist_ok=True)

    # move the files
    for f in train_files:
        os.rename(path + f, path + "train/" + f)
    for f in valid_files:    
        os.rename(path + f, path + "valid/" + f)
    for f in test_files:
        os.rename(path + f, path + "test/" + f)

if __name__ == "__main__":

    cwd = os.getcwd()
    path = os.path.join(cwd, "data/raw/eval_images/")




    split(0.8, 0.1, 0.1, path)