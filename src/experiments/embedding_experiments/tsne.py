from sklearn.manifold import TSNE

import sys
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import pandas as pd 
import pickle
import matplotlib.pyplot as plt

sys.path.append('../../')

import util.embedding_utils as embedding_utils
import util.util as util
import dataset.dataloader as dataloader
from dataset.tasks import Task, elevation_task, roads_task, treecover_task
from dataset.dataloader import EmbeddedDataset

def plot_tsne(dataset, filename):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    X = dataset.X
    y = dataset.y 

   # create new Pandas dataframe
    df = pd.DataFrame()
    df['y'] = y

    tsne_results = tsne.fit_transform(X)

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    # get the range of the values in df['y']
    y_range = df['y'].max() - df['y'].min()
    print("Y RANGE ", y_range)

    # TODO: Re-run bins with elevation here as well!
#    bins = np.array([0, int(0.25 * y_range), int(0.5 * y_range), int(0.75 * y_range), y_range])
    
#    if (task == 'elevation'):
    bins = np.array([0, 1000, 2000, 3000, 4000])
#    elif (task == 'treecover'):
#        bins = np.array([0, 25, 50, 75, 100])

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

    sns_plot.figure.savefig("plots/tsne/" + filename + ".png")

if __name__ == "__main__":
    # load embeddings 
    # what embeddings to train on/generate visualizations for?? Possibly for the four tasks model??
    model_name = "pretrained_visiontransformer_ElInPoRdTrNl"

    with open('../../out/embeddings/' + util.get_embedding_filename(model_name), 'rb') as f:
        embeddings = pickle.load(f)

    with open('./../../../data/int/CONTUS_UAR.pkl', 'rb') as f:
        mosaiks_embeddings = pickle.load(f)
        X = mosaiks_embeddings["X"]
        ids_X = mosaiks_embeddings["ids_X"]
        mosaiks_embeddings = embedding_utils.mosaiks_format_to_map(X, ids_X, embeddings)

    embedded_dataset_elevation = EmbeddedDataset(embeddings, elevation_task)
    embedded_dataset_mosaiks_elevation = EmbeddedDataset(mosaiks_embeddings, elevation_task)

    embedded_dataset_treecover = EmbeddedDataset(embeddings, treecover_task)
    embedded_dataset_mosaiks_treecover = EmbeddedDataset(embeddings, treecover_task)

    plot_tsne(embedded_dataset_elevation, "tsne_elevation_2")

# Plot already generated: stored in plots/tsne_elevation.png
#    plot_tsne(embedded_dataset_elevation, "embeddings elevation")

#    plot_tsne(embedded_dataset_mosaiks_elevation, "mosaiks_embeddings")