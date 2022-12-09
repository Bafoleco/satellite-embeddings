import scipy.stats
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import pickle

import sys
sys.path.append('../..')

import dataset.tasks as tasks
import dataset.dataloader as dataloader
import util.embedding_utils as embedding_utils
import util.util as util
import eval_stage.regression as regression
import seaborn as sns
import pandas as pd


if __name__ == "__main__":
    # load embeddings 
    model_name = "pretrained_visiontransformer_512_ElRdInTr"

    with open(os.path.join(util.get_embeddings_path(), util.get_embedding_filename(model_name)), 'rb') as f:
        embeddings = pickle.load(f)

    print("Ours:")


    our_results = {}

    for task in tasks.acs_tasks:

        print(task.name)

        # construct dataset
        dataset = dataloader.EmbeddedDataset(embeddings, task)

        # split into train and eval
        train_X, train_y, valid_X, valid_y, test_X, test_y = dataset.split()
        # print("Train size: ", len(train_X))

        dir = os.path.join(".", "plots", model_name)
        if not os.path.exists(dir):
            os.makedirs(dir)

        score = regression.train_and_eval(train_X, train_y, valid_X, valid_y, task.name, dir, lamb=0)
        our_results[task.name] = score

    with open(os.path.join(util.get_data_path(), "int", "CONTUS_UAR.pkl"), 'rb') as f:
        mosaiks_embeddings = pickle.load(f)
        X = mosaiks_embeddings["X"]
        ids_X = mosaiks_embeddings["ids_X"]
        mosaiks_embeddings = embedding_utils.mosaiks_format_to_map(X, ids_X, embeddings)

    print("MOSAIKS:")

    mosaiks_results = {}

    for task in tasks.acs_tasks:

        print(task.name)

        dataset = dataloader.EmbeddedDataset(mosaiks_embeddings, task)

        # split into train and eval
        train_X, train_y, valid_X, valid_y, test_X, test_y = dataset.split()

        dir = os.path.join(".", "plots", model_name + "_mosaiks")
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        score = regression.train_and_eval(train_X, train_y, valid_X, valid_y, task.name, dir)
        mosaiks_results[task.name] = score
    
    # create bargraph with seaborn
    df = pd.DataFrame.from_dict({"Ours": our_results, "MOSAIKS": mosaiks_results})

    # df = pd.DataFrame.from_dict({"Ours": our_results})

    # df = pd.DataFrame.from_dict({"MOSAIKS": mosaiks_results})

    df = df.transpose()

    # replace col names using task map and display names
    df.columns = [tasks.task_name_map[task].display_name for task in df.columns]

    # reset index

    df.reset_index(inplace=True)

    # rename index column

    df.rename(columns={"index": "Model"}, inplace=True)


    print(df)

    # reshape to put score as a column
    df = pd.melt(df, id_vars=["Model"], var_name="Task", value_name="Score")

    print(df)

    # seaborn barplot

    # better colors for the barplot
    sns.set_palette("Set2")

    ax = sns.barplot(data=df, hue="Model", x="Task", y="Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    # plt.show()

    # t

    dir = os.path.join(".", "plots", model_name)


    plt.savefig(os.path.join(dir, "barplot.png"))

