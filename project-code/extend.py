import torch # PyTorch package
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt



def embed_images(model):


    # https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/05/27/extracting-features.html
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook


    model = None



    features = {}

    model.global_pool.register_forward_hook(get_features('feats'))

    dataset = util.create_dataset_home_prices(transfrom)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    with torch.no_grad():
        for images, labels in data_loader:
            model(images)
            