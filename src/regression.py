import scipy.stats
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import pickle
import util
import dataloader
import tasks
import util
import embedding_utils

def train_and_eval(train_X, train_y, eval_X, eval_y):
    """
    Train and evaluate a linear regression model
    """
    model = Ridge(0.01).fit(train_X, train_y)
    score = model.score(eval_X, eval_y)
    print("Score: " + str(score))
    print("Cross Validation Score: " + str(cross_val_score(model, eval_X, eval_y, cv=5)))

    y_pred = model.predict(eval_X)
    print("Mean Absolute Error: ", mean_absolute_error(eval_y, y_pred))
    print("Mean Squared Error: ", mean_squared_error(eval_y, y_pred))

    return score

if __name__ == "__main__":
    # load embeddings 
    model_name = "pretrained_visiontransformer_ElInPoRdTrNl"

    training_tasks = embedding_utils.parse_tasks(model_name)

    with open('./embeddings/' + util.get_embedding_filename(model_name), 'rb') as f:
        embeddings = pickle.load(f)

    with open('../data/int/CONTUS_UAR.pkl', 'rb') as f:
        mosaiks_embeddings = pickle.load(f)
        X = mosaiks_embeddings["X"]
        ids_X = mosaiks_embeddings["ids_X"]
        mosaiks_embeddings = embedding_utils.mosaiks_format_to_map(X, ids_X, embeddings)

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
