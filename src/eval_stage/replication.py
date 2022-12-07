import scipy.stats
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

import pickle
import util.util as util
import dataset.dataloader as dataloader
import dataset.tasks as tasks

# we attempt to validate the MOSAIKS results using our own code


def mosaiks_format_to_map(X, ids_X, dim=2048):
    """
    Convert the mosaiks format to a map
    """
    mosaiks_map = {}
    for i in range(len(ids_X)):
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
    with open('../data/int/CONTUS_UAR.pkl', 'rb') as f:
        mosaiks_embeddings = pickle.load(f)
        X = mosaiks_embeddings["X"]
        ids_X = mosaiks_embeddings["ids_X"]
        mosaiks_embeddings = mosaiks_format_to_map(X, ids_X, dim=2048)

    for task in tasks.all_tasks:
        print("Comparing on", task.name)
        dataset = dataloader.EmbeddedDataset(mosaiks_embeddings, task)

        train_X, train_y, valid_X, valid_y, test_X, test_y = dataset.split()
        train_and_eval(train_X, train_y, valid_X, valid_y)
