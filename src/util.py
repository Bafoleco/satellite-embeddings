from operator import le
import numpy as np
import torch # PyTorch package
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import scipy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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


def graph_performance(data_loader, dataset, net):
    pred_map = init_pred_map(dataset)

    with torch.no_grad():
        for inputs, labels, _ in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            save_predictions(dataset, labels, outputs, pred_map)

    for task in dataset.tasks:
        draw_graph(task, pred_map)

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

def draw_graph(task, scatter_data):
    x, y = scatter_data[task.name]
    plt.close()
    plt.scatter(x, y)
    plt.xlabel("true")
    plt.ylabel("predicted")
    plt.title(task.display_name)
    plt.savefig("plots/" + task.name + ".png")


def get_percent_error(outputs, labels):
    # print(outputs.shape)
    # print(labels.shape)

    return np.average(np.abs(outputs - labels) / labels) * 100
