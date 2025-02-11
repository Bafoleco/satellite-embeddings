import numpy as np
import torch # PyTorch package
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os 
from pathlib import Path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def image_present(root_dir, indices):
    ij = indices.split(",")
    i = ij[0]
    j = ij[1]

    file_name = str(i) + "_" + str(j) + ".png"
    return os.path.exists(os.path.join(root_dir, file_name))

def get_embedding_filename(model_name):
    return model_name + "_embeddings.pkl"

def get_model_loss(data_loader, dataset, net, loss_function):
    pred_map = init_pred_map(dataset) 

    with torch.no_grad():
        total_loss = 0
        batches = 0        
        for inputs, labels, _ in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            save_predictions(dataset, labels, outputs, pred_map)

            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            batches += 1
            

    for task in dataset.tasks:
        # print r2 for task
        x, y = pred_map[task.name]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

        print("R2 for " + task.name + " is " + str(r_value))
        
    return total_loss / batches

def graph_performance(plot_dir, data_loader, dataset, net):
    pred_map = init_pred_map(dataset)

    with torch.no_grad():
        for inputs, labels, _ in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            save_predictions(dataset, labels, outputs, pred_map)

    for task in dataset.tasks:
        draw_graph(plot_dir, task, pred_map)

def init_pred_map(dataset):
    tasks = dataset.tasks
    pred_map = {}
    for task in tasks:
        pred_map[task.name] = ([], [])

    return pred_map

def save_predictions(dataset, labels, outputs, scatter_data):
    predictions = dataset.transform_output(outputs.cpu().detach())
    true = dataset.transform_output(labels.cpu())

    for i, task in enumerate(dataset.tasks):
        x, y = scatter_data[task.name]
        for j in range(len(predictions)):
            x.append(predictions[j][i])
            y.append(true[j][i])

def draw_graph(plot_dir, task, scatter_data):
    x, y = scatter_data[task.name]
    plt.close()
    plt.scatter(x, y)
    plt.xlabel("true")
    plt.ylabel("predicted")
    plt.title(task.display_name)

    # If the plot directory doesn't exist, make it
    if (not os.path.exists(plot_dir)):
        os.mkdir(plot_dir)

    plt.savefig(plot_dir + task.name + ".png")


def get_percent_error(outputs, labels):
    # print(outputs.shape)
    # print(labels.shape)

    return np.average(np.abs(outputs - labels) / labels) * 100

source_path = Path(__file__).resolve()
source_dir = source_path.parent

def get_models_path():
    return os.path.join(source_dir.parent, "out", "models")

def get_embeddings_path():
    return os.path.join(source_dir.parent, "out", "embeddings")

def get_data_path():
    return os.path.join(source_dir.parent.parent, "data")

def get_eval_images_path():
    return os.path.join(get_data_path(), "raw", "eval_images")