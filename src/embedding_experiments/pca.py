import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

import pickle
import matplotlib.pyplot as plt

sys.path.append('../src')

import embedding_utils
import util

def plot_pca(X, name):
    sc = StandardScaler()
    sc.fit(X)
    X_std = sc.transform(X)

    pca = PCA(n_components=50)
    pca.fit(X_std)

    # print expalined variance ratio
    print("Explained variance ratio: ", pca.explained_variance_ratio_)

    # plot explained variance ratio
    plt.plot(pca.explained_variance_ratio_)

    # plot cumulative
    plt.plot(np.cumsum(pca.explained_variance_ratio_))

    plt.xlabel("Component")
    plt.ylabel("Explained Variance Ratio")


    plt.savefig("embedding_experiments/plots/" + name + "_pca" + ".png")

    plt.close()


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


    embeddings, _ = embedding_utils.convert_map_to_nparray(embeddings)
    mosaiks_embeddings, _ = embedding_utils.convert_map_to_nparray(mosaiks_embeddings)

    plot_pca(embeddings, "embeddings")
    plot_pca(mosaiks_embeddings, "mosaiks_embeddings")

