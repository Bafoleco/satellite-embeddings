import scipy.stats
import numpy as np
from sklearn.linear_model import LinearRegression

import pickle
import util
import dataloader
import tasks


# use MOSAIKS

# load using pickle



def mosaiks_format_to_map(X, ids_X):
    """
    Convert the mosaiks format to a map
    """
    mosaiks_map = {}
    for i in range(len(ids_X)):
        mosaiks_map[ids_X[i]] = X[i]
    return mosaiks_map


def train_and_eval(train_X, train_y, eval_X, eval_y):
    """
    Train and evaluate a linear regression model
    """
    model = LinearRegression().fit(train_X, train_y)
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
        mosaiks_embeddings = mosaiks_format_to_map(X, ids_X)

    for task in tasks.all_tasks:
        print("Comparing on ", task.name)

        # construct dataset
        dataset = dataloader.EmbeddedDataset(embeddings, task)
        mosaiks_dataset = dataloader.EmbeddedDataset(mosaiks_embeddings, task)

        # split into train and eval
        train_X, train_y, valid_X, valid_y, test_X, test_y = dataset.split()

        train_and_eval(train_X, train_y, valid_X, valid_y)

        train_X, train_y, valid_X, valid_y, test_X, test_y = mosaiks_dataset.split()

        train_and_eval(train_X, train_y, valid_X, valid_y)

