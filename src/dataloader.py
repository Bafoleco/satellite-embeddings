import torchvision # load datasets
import torchvision.transforms as transforms # transform data
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import numpy as np

# python image library of range [0, 1] 
# transform them to tensors of normalized range[-1, 1]
base_transform = transforms.Compose( # composing several transforms together
    [torchvision.transforms.Resize(256),
     transforms.ToTensor(), # to tensor object
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean = 0.5, std = 0.5

def image_present(root_dir, indices):
    ij = indices.split(",")
    i = ij[0]
    j = ij[1]

    file_name = str(i) + "_" + str(j) + ".png"
    return os.path.exists(os.path.join(root_dir, file_name))


def normalize(col):
    col_mean = col.mean()
    col_std = col.std()
    norm_col = (col - col_mean) / col_std
    return norm_col, (col_mean, col_std)
    

class SatDataset(Dataset):

    def transform_output(self, output, label="price"):

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
            print(task)               
            csv = pd.read_csv(task.csv_file)

            mask = csv.apply(lambda row: image_present(root_dir, row["ID"]), axis=1)

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
        
        # load image
        id = self.labels.iloc[i, self.labels.columns.get_loc("ID")]

        file_name = id.replace(",", "_") + ".png"
        img_name = os.path.join(self.root_dir, file_name)
        image = Image.open(img_name).convert('RGB')

        labels = self.labels.iloc[i, 1:].to_numpy().astype(np.float32)
 
        return [self.transform(image), labels, id]

class EmbeddedDataset:

    def __init__(self, embeddings, task):

        print("init embedded dataset")

        # get key set of embeddings
        keys = set(embeddings.keys())
        dim = len(embeddings[next(iter(keys))])

        csv = pd.read_csv(task.csv_file)

        # print("csv", csv.head())

        # print("keys", keys)

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
        self.y = y