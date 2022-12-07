import torchvision # load datasets
import torchvision.transforms as transforms # transform data
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import numpy as np
import util

# python image library of range [0, 1] 
# transform them to tensors of normalized range[-1, 1]
base_transform = transforms.Compose( # composing several transforms together
    [torchvision.transforms.Resize(256),
     transforms.ToTensor(), # to tensor object
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean = 0.5, std = 0.5

def normalize(col):
    col_mean = col.mean()
    col_std = col.std()
    norm_col = (col - col_mean) / col_std
    return norm_col, (col_mean, col_std)
    
class SatDataset(Dataset):

    def transform_output(self, output):

        transformed_output = np.zeros(output.shape)
        for i in range(output.shape[1]):
            task = self.tasks[i]
            # undo normalization
            transformed_output[:, i] = output[:, i] * self.label_transforms[task][1] + self.label_transforms[task][0]

        return transformed_output

    def __init__(self, tasks, root_dir, transform=None):
        transform = base_transform if transform is None else transform

        self.tasks = tasks

        # init label transform map
        self.label_transforms = {}

        df = pd.DataFrame()
        
        for task in tasks:
            print("Root dir ", root_dir)
            print("Adding images for task ", task.name)               
            csv = pd.read_csv(task.csv_file)

            mask = csv.apply(lambda row: util.image_present(root_dir, row["ID"]), axis=1)

            filtered_csv = csv[mask]

            filtered_csv.set_index("ID", inplace=True)

            col = filtered_csv.loc[:, task.col_name]
            norm_col, col_transform = normalize(col)

            self.label_transforms[task] = col_transform

            df.insert(len(df.columns), task.col_name, norm_col)
            print("df", df)
            
        self.labels = df.reset_index()

        print(self.labels.head())
        
        # set image parameters
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        id = self.labels.iloc[i, self.labels.columns.get_loc("ID")]

        file_name = id.replace(",", "_") + ".png"
        img_name = os.path.join(self.root_dir, file_name)
        image = Image.open(img_name).convert('RGB')

        labels = self.labels.iloc[i, 1:].to_numpy().astype(np.float32)
 
        return [self.transform(image), labels, id]

    def get_task_code(self):
        code = ""
        for task in self.tasks:
            code += task.code
        return code
        
class EmbeddedDataset:

    def __init__(self, embeddings, task):
        # print("init embedded dataset")
        # get key set of embeddings
        keys = set(embeddings.keys())
        dim = len(embeddings[next(iter(keys))])

        csv = pd.read_csv(task.csv_file)

        mask = csv.apply(lambda row: row["ID"] in keys, axis=1)

        filtered_csv = csv[mask]
        filtered_csv = filtered_csv.reset_index()

        # print("size: ")
        # print(len(filtered_csv))

        X = np.zeros((len(filtered_csv), dim))
        y = np.zeros((len(filtered_csv), 1))

        for i in range(len(filtered_csv)):
            id = filtered_csv.iloc[i, filtered_csv.columns.get_loc("ID")]
            X[i, :] = embeddings[id]
            y[i] = filtered_csv.iloc[i, filtered_csv.columns.get_loc(task.col_name)]

        # print("X", X)

        self.X = X
        self.y = y[:,0]


    # TODO shuffling, but like deterministic shuffling
    # train valid test split
    def split(self, train_size=0.8, valid_size=0.1, test_size=0.1):
        assert train_size + test_size + valid_size == 1

        train_size = int(train_size * len(self.X))
        valid_size = int(valid_size * len(self.X))
        test_size = len(self.X) - train_size - valid_size

        train_X = self.X[:train_size]
        train_y = self.y[:train_size]

        valid_X = self.X[train_size:train_size + valid_size]
        valid_y = self.y[train_size:train_size + valid_size]

        test_X = self.X[train_size + valid_size:]
        test_y = self.y[train_size + valid_size:]

        return train_X, train_y, valid_X, valid_y, test_X, test_y