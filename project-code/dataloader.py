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
        
        for i, task in enumerate(tasks): 
            print(task)               
            csv_file, column_title = task
            csv = pd.read_csv(csv_file)
            print(len(csv))
            mask = csv.apply(lambda row: image_present(root_dir, row["ID"]), axis=1)
            print(len(mask))


            print("mask", mask)

            filtered_csv = csv[mask]

            if i == 0:
                id_col = filtered_csv.loc[:, "ID"]
                df.insert(len(df.columns), "ID", id_col)

            col = filtered_csv.loc[:, column_title]

            norm_col, col_transform = normalize(col)
            self.label_transforms[task] = col_transform

            df.insert(len(df.columns), column_title, norm_col)
            

        self.labels = df
        
        # set image parameters
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        
        # load image
        file_name = self.labels.iloc[i, self.labels.columns.get_loc("ID")].replace(",", "_") + ".png"
        img_name = os.path.join(self.root_dir, file_name)
        image = Image.open(img_name).convert('RGB')


        labels = self.labels.iloc[i, 1:].to_numpy().astype(np.float32)
 
        return [self.transform(image), labels]
