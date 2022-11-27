import scipy.stats
import numpy as np
from sklearn.linear_model import LinearRegression

import pickle
import util
import dataloader
import tasks


# use MOSAIKS

# load using pickle
with open('../data/int/CONTUS_UAR.pkl', 'rb') as f:
    mosaiks_embeddings = pickle.load(f)


    X = mosaiks_embeddings["X"]
    ids_X = mosaiks_embeddings["ids_X"]

    print(X.shape)


# load embeddings 
model_name = "pretrained_resnet"

with open('./embeddings/' + util.get_embedding_filename(model_name), 'rb') as f:
    embeddings = pickle.load(f)

for task in tasks.all_tasks:
    print(task.name)

    # construct dataset
    dataset = dataloader.EmbeddedDataset(embeddings, task)

    X = dataset.X
    y = dataset.y[:,0]

    print(X.shape)
    print(y.shape)


    reg = LinearRegression().fit(X, y)

    print(reg.score(X, y))


    