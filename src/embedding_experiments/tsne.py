from sklearn.manifold import TSNE

import sys
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import pandas as pd 
import pickle
import matplotlib.pyplot as plt

sys.path.append('../src')

import embedding_utils
import util
import dataloader
from tasks import Task, elevation_task, roads_task
from dataloader import EmbeddedDataset

def plot_tsne(dataset, name):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    X = dataset.X
    y = dataset.y 

   # create new Pandas dataframe
    df = pd.DataFrame()
#    df['X'] = X
    df['y'] = y

    tsne_results = tsne.fit_transform(X)

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    # get the range of the values in df['y']
    y_range = df['y'].max() - df['y'].min()
    print("Y RANGE ", y_range)

    bins = np.array([0, 1000, 2000, 3000, 4000])
    inds = np.digitize(df['y'], bins)
    df['y'] = inds

    y_range = df['y'].max() - df['y'].min()
    print("Y RANGE ", y_range)

    sns_plot = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 5),
        data=df,
        legend="full",
        alpha=0.3
    )

    sns_plot.figure.savefig("plots/tsne_elevation.png")

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

    embedded_dataset_elevation = EmbeddedDataset(embeddings, elevation_task)

    plot_tsne(embedded_dataset_elevation, "embeddings elevation")

#    plot_tsne(mosaiks_embeddings, "mosaiks_embeddings")