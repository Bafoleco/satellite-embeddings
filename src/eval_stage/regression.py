import scipy.stats
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

import sys
sys.path.append('../')

import pickle
import dataset.dataloader as dataloader
import dataset.tasks as tasks
import util.embedding_utils as embedding_utils
import util.util as util
import seaborn as sns 

def draw_graph(y_pred, y, task_name, dir):
    """
    Draw a graph of the predictions
    """
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(task_name)

    scatter_color = "green"
    line_color = "gray"

    if (task_name == 'population'):
        scatter_color = "darkcyan"
    elif (task_name == 'nightlights'):
        scatter_color = "limegreen"

    sns.regplot(y, y_pred, scatter_kws={"color": scatter_color, "s": 5}, line_kws={"color": "red"}, ci = None, fit_reg = False)
    ax = plt.gca()
    ax.axline((0, 0), slope=1, color=line_color)

    print("Saving figure")

    plt.savefig(os.path.join(dir, task_name + "_plotexp" + ".png"))
    plt.clf()

def count_zeros(X):
    """
    Count the number of zeros in a matrix
    """
    count = 0
    total = 0
    for row in X:
        for elem in row:
            total += 1
            if elem == 0:
                count += 1
    return count / total

def count_all_zero_columns(X):
    """
    Count the number of columns that are all zeros
    """
    count = 0
    for i in range(len(X[0])):
        if np.count_nonzero(X[:, i]) == 0:
            count += 1
    return count

def train_lin_reg(params):
    model = Ridge(params['ridge_coeff']).fit(params['train_X'], params['train_y'])
    return {'loss': model.score(params['eval_X'], params['eval_y']), 'status': STATUS_OK}

def train_and_eval(train_X, train_y, eval_X, eval_y, taskname, dir, lamb=2):
    """
    Train and evaluate a linear regression model
    """

    print("Train size: ", len(train_X))
    print("Dimensions: ", len(train_X[0]))

    model = Ridge(lamb).fit(train_X, train_y)

    fspace = {
         'ridge_coeff': hp.lognormal('ridge_coeff', 0.01, 0.5),
         'train_X': train_X,
         'train_y': train_y,
         'eval_X': eval_X,
         'eval_y': eval_y,
    }

    trials = Trials()

    best = fmin(
            fn=train_lin_reg,
            space=fspace,
            algo=tpe.suggest,
            max_evals=150)

    print("Best: ", best)
    model = Ridge(best['ridge_coeff']).fit(train_X, train_y)

    score = model.score(eval_X, eval_y)

    print("Score: " + str(score))

    y_pred = model.predict(eval_X)
    print("Mean Absolute Error: ", mean_absolute_error(eval_y, y_pred))
    print("Mean Squared Error: ", mean_squared_error(eval_y, y_pred))

    # graph
    draw_graph(y_pred, eval_y, taskname, dir)

    return score

if __name__ == "__main__":
    # load embeddings 
    model_name = "pretrained_visiontransformer_1024_ElRdIn"
    # model_name = "pretrained_visiontransformer_1024_ElRdIn"

    full_model_name = "pretrained_visiontransformer_1024_ElRdInTr"
    training_tasks = embedding_utils.parse_tasks(full_model_name)
    # training_tasks = []

    with open(os.path.join(util.get_embeddings_path(), util.get_embedding_filename(model_name)), 'rb') as f:
        embeddings = pickle.load(f)

    print("Ours:")

    for task in tasks.all_tasks:
        if task in training_tasks:
            continue

        print(task.name)

        # construct dataset
        dataset = dataloader.EmbeddedDataset(embeddings, task)

        # split into train and eval
        train_X, train_y, valid_X, valid_y, test_X, test_y = dataset.split()
        # print("Train size: ", len(train_X))

        dir = os.path.join(".", "plots", model_name)
        if not os.path.exists(dir):
            os.makedirs(dir)

        train_and_eval(train_X, train_y, test_X, test_y, task.name, dir)

    with open('../../data/int/CONTUS_UAR.pkl', 'rb') as f:
        mosaiks_embeddings = pickle.load(f)
        X = mosaiks_embeddings["X"]
        ids_X = mosaiks_embeddings["ids_X"]
        mosaiks_embeddings = embedding_utils.mosaiks_format_to_map(X, ids_X, embeddings)

    print("MOSAIKS:")

    for task in tasks.all_tasks:
        if task in training_tasks:
            continue

        print(task.name)

        dataset = dataloader.EmbeddedDataset(mosaiks_embeddings, task)

        # split into train and eval
        train_X, train_y, valid_X, valid_y, test_X, test_y = dataset.split()

        dir = os.path.join(".", "plots", model_name + "_mosaiks")
        if not os.path.exists(dir):
            os.makedirs(dir)
        train_and_eval(train_X, train_y, test_X, test_y, task.name, dir)
