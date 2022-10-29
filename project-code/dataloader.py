import torchvision # load datasets
import torchvision.transforms as transforms # transform data
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import numpy as np


# python image library of range [0, 1] 
# transform them to tensors of normalized range[-1, 1]
transform = transforms.Compose( # composing several transforms together
    [torchvision.transforms.Resize(256),
     transforms.ToTensor(), # to tensor object
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean = 0.5, std = 0.5

def image_present(root_dir, indices):

    # print(indices)

    ij = indices.split(",")
    i = ij[0]
    j = ij[1]

    file_name = str(i) + "_" + str(j) + ".png"
    return os.path.exists(os.path.join(root_dir, file_name))




uar_elevation = "outcomes_sampled_elevation_CONTUS_16_640_UAR_100000_0.csv"
uar_income = "outcomes_sampled_income_CONTUS_16_640_UAR_100000_0.csv"
uar_population = "outcomes_sampled_population_CONTUS_16_640_UAR_100000_0.csv"
uar_roads = "outcomes_sampled_roads_CONTUS_16_640_UAR_100000_0.csv"
uar_treecover = "outcomes_sampled_treecover_CONTUS_16_640_UAR_100000_0.csv"
uar_nightlights = "outcomes_sampled_nightlights_CONTUS_16_640_UAR_100000_0.csv"


def normalize(y):
    mean = np.average(y)
    std = np.std(y)
    # print(std)
    return (y - mean) / std, mean, std

class SatDataset(Dataset):

    def transform_output(self, output, label="price"):
        mean, std = self.label_transforms[label]
        return std * output + mean

    def __init__(self, csv_file, root_dir, elevation=False, income=False, pop=False, roads=False, treecover=False, lights=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        all_labels = pd.read_csv(csv_file)
        
        # filter missing
        mask = all_labels.apply(lambda row: image_present(root_dir, row[0]), axis=1)
        labels = all_labels[mask]

        # init label transform map
        self.label_transforms = {}

        
        # normalize price col
        price_col = labels.loc[:, "price"]

        mean_price = price_col.mean()
        std_price = price_col.std()

        self.label_transforms["price"] = (mean_price, std_price)

        norm_prices = (price_col - price_col.mean()) / price_col.std()
        labels.loc[:, "price"] = norm_prices

        self.labels = labels

        
        # set image parameters
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        i = idx

        # for i in idx:

        # print(self.labels.iloc[i])

        file_name = self.labels.iloc[i, 0].replace(",", "_") + ".png"

        # print(file_name)

        img_name = os.path.join(self.root_dir,
                                file_name)
        image = Image.open(img_name).convert('RGB')

    
        price = self.labels.iloc[idx, 1]
        # prices.append(price)
        # images.append(self.transform(image))


        return [self.transform(image), price.astype(np.float32)]
