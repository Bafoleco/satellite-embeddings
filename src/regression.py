import scipy.stats
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

import pickle
import util
import dataloader
import tasks
import util

# use MOSAIKS

# load using pickle

def mosaiks_format_to_map(X, ids_X, embeddings, dim=2048):
    """
    Convert the mosaiks format to a map
    """
    mosaiks_map = {}
    for i in range(len(ids_X)):
        if ids_X[i] in embeddings:
            mosaiks_map[ids_X[i]] = X[i][:dim]
    return mosaiks_map


def train_and_eval(train_X, train_y, eval_X, eval_y):
    """
    Train and evaluate a linear regression model
    """
    model = Ridge(0.01).fit(train_X, train_y)
    score = model.score(eval_X, eval_y)
    print("Score: " + str(score))
    return score

if __name__ == "__main__":
    # load embeddings 
    model_name = "pretrained_resnet"

    with open('./embeddings/' + util.get_embedding_filename(model_name), 'rb') as f:
        embeddings = pickle.load(f)

    with open('../data/int/CONTUS_UAR.pkl', 'rb') as f:
        mosaiks_embeddings = pickle.load(f)
        X = mosaiks_embeddings["X"]
        ids_X = mosaiks_embeddings["ids_X"]
        mosaiks_embeddings = mosaiks_format_to_map(X, ids_X, embeddings)

    for task in tasks.all_tasks:
        print("Comparing on ", task.name)

        # construct dataset
        dataset = dataloader.EmbeddedDataset(embeddings, task)

        # split into train and eval
        train_X, train_y, valid_X, valid_y, test_X, test_y = dataset.split()
        # print("Train size: ", len(train_X))

        print("Ours:")
        train_and_eval(train_X, train_y, valid_X, valid_y)

        dataset = dataloader.EmbeddedDataset(mosaiks_embeddings, task)

        # split into train and eval
        train_X, train_y, valid_X, valid_y, test_X, test_y = dataset.split()

        print("MOSAIKS:")
        train_and_eval(train_X, train_y, valid_X, valid_y)
