import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import shutil
import os

import pickle
import matplotlib.pyplot as plt

sys.path.append('../src')

import embedding_utils
import util


def interpolate(start, end, embeddings, steps=15, mosaiks=False):
    X, ids_X = embedding_utils.convert_map_to_nparray(embeddings)
    nn = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X)

    start_embedding = embeddings[start]
    end_embedding = embeddings[end]

    diff = end_embedding - start_embedding

    # create directory for interpolation
    print(mosaiks)
    interpolation_dir = os.path.join("embedding_experiments", "interpolation", start + "_" + end + ("_mosaiks" if mosaiks else ""))

    # remove directory if it exists
    if os.path.exists(interpolation_dir):
        shutil.rmtree(interpolation_dir)

    os.makedirs(interpolation_dir)

    # interpolate
    interpolated = []
    for i in range(steps + 1):
        intermediate = start_embedding + (i / steps) * diff

        # find nearest neighbors
        distances, indices = nn.kneighbors([intermediate])

        # get ids
        ids = [ids_X[index] for index in indices]

        # create folder for this step
        step_folder = os.path.join(interpolation_dir, "step_" + str(i))

        print(step_folder)

        os.makedirs(step_folder)

        # save images
        for id in ids[0]:
            # copy file from ../data/raw to step_folder
            id = id.replace(",", "_")
            shutil.copyfile("../data/raw/eval_images/" + id + ".png", step_folder + "/" + id + ".png")
        


def save_neigbors(id, embeddings, n_neighbors, mosaiks=False):
    X, ids_X = embedding_utils.convert_map_to_nparray(embeddings)
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(X)

    # create directory for interpolation
    neighbors_dir = os.path.join("embedding_experiments", "neighbors", ("mosaiks" if mosaiks else ""))

    # remove directory if it exists
    if os.path.exists(neighbors_dir):
        shutil.rmtree(neighbors_dir)

    os.makedirs(neighbors_dir)

    # find nearest neighbors
    distances, indices = nn.kneighbors([embeddings[id]])

    # get ids
    ids = [ids_X[index] for index in indices]

    # create folder for this step
    id_folder = os.path.join(neighbors_dir, id)

    print(id_folder)

    os.makedirs(id_folder)

    # save images
    for id in ids[0]:
        # copy file from ../data/raw to step_folder
        id = id.replace(",", "_")
        shutil.copyfile("../data/raw/eval_images/" + id + ".png", id_folder + "/" + id + ".png")


if __name__ == "__main__":
    # load embeddings 
    model_name = "pretrained_visiontransformer_ElInPoRdTrNl"

    with open('./embeddings/' + util.get_embedding_filename(model_name), 'rb') as f:
        embeddings = pickle.load(f)

    with open('../data/int/CONTUS_UAR.pkl', 'rb') as f:
        mosaiks_embeddings = pickle.load(f)
        X = mosaiks_embeddings["X"]
        ids_X = mosaiks_embeddings["ids_X"]
        mosaiks_embeddings = embedding_utils.mosaiks_format_to_map(X, ids_X, embeddings)


    # interpolate
    # wilderness to urban
    interpolate("116,1583", "1961,1930", embeddings)
    interpolate("116,1583", "1961,1930", mosaiks_embeddings, mosaiks=True)


    # X, ids_X = embedding_utils.convert_map_to_nparray(embeddings)
    # mosaiks_embeddings, _ = embedding_utils.convert_map_to_nparray(mosaiks_embeddings)

    save_neigbors("116,1583", embeddings, 5)
    save_neigbors("116,1583", mosaiks_embeddings, 5, mosaiks=True)


    # search_vec = embeddings[0]

    # ## search for nearest neighbors
    # # get nearest neighbors
    # nn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(embeddings)
    # distances, indices = nn.kneighbors([search_vec])


    # print(ids_X[indices])


