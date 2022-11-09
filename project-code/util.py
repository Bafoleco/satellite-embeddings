from operator import le
import numpy as np
from dataloader import SatDataset
import torch # PyTorch package
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import scipy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

base = "../data/int/applications"
uar_elevation = base + "/elevation/outcomes_sampled_elevation_CONTUS_16_640_UAR_100000_0.csv"
uar_income = base + "/income/outcomes_sampled_income_CONTUS_16_640_UAR_100000_0.csv"
uar_population = base + "outcomes_sampled_population_CONTUS_16_640_UAR_100000_0.csv"
uar_roads = base + "outcomes_sampled_roads_CONTUS_16_640_UAR_100000_0.csv"
uar_treecover = base + "/treecover/outcomes_sampled_treecover_CONTUS_16_640_UAR_100000_0.csv"
uar_nightlights = base + "outcomes_sampled_nightlights_CONTUS_16_640_UAR_100000_0.csv"

elevation_title = "elevation"
income_title = "income"
population_title = "population"
roads_title = "roads"
treecover_title = "treecover"
nightlights_title = "nightlights"

image_root = "../data/raw/mosaiks_images"


def create_dataset_all(transfrom):
    tasks = [(uar_elevation, elevation_title), 
             (uar_income, income_title), 
             (uar_population, population_title), 
             (uar_roads, roads_title), 
             (uar_treecover, treecover_title), 
             (uar_nightlights, nightlights_title)]
    return SatDataset(tasks, image_root, transfrom)

def create_dataset_treecover(transfrom):
    tasks = [(uar_treecover, treecover_title)]
    return SatDataset(tasks, image_root, transfrom)


def create_dataset_income(transfrom):
    tasks = [(uar_income, income_title)]
    return SatDataset(tasks, image_root, transfrom)

# def create_dataset_home_prices(transfrom):
#     tasks = [("../data/int/applications/housing/outcomes_sampled_housing_CONTUS_16_640_POP_100000_0.csv", "price")]
#     return SatDataset(tasks, image_root, transfrom)


def get_model_loss(data_loader, dataset, net, loss_function):
    pred_map = init_pred_map(dataset) 

    with torch.no_grad():
        total_loss = 0
        batches = 0        
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            save_predictions(dataset, labels, outputs, pred_map)

            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            batches += 1
            

    for task in dataset.tasks:
        # print r2 for task
        x, y = pred_map[task]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

        print("R2 for " + task[1] + " is " + str(r_value))
        
    return total_loss / batches


def graph_performance(data_loader, dataset, net):
    pred_map = init_pred_map(dataset)

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            save_predictions(dataset, labels, outputs, pred_map)

    for task in dataset.tasks:
        draw_graph(task, pred_map)

def init_pred_map(dataset):
    tasks = dataset.tasks
    pred_map = {}
    for task in tasks:
        pred_map[task] = ([], [])

    return pred_map

def save_predictions(dataset, labels, outputs, scatter_data):
    predictions = dataset.transform_output(outputs.cpu().detach())
    true = dataset.transform_output(labels.cpu())

    for i, task in enumerate(dataset.tasks):
        x, y = scatter_data[task]
        for j in range(len(predictions)):
            x.append(predictions[j][i])
            y.append(true[j][i])

def draw_graph(task, scatter_data):
    _, title = task
    x, y = scatter_data[task]
    plt.close()
    plt.plot(x, y)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.savefig(title + ".png")


def get_percent_error(outputs, labels):
    # print(outputs.shape)
    # print(labels.shape)

    return np.average(np.abs(outputs - labels) / labels) * 100
