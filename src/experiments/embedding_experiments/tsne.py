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
from dataset.tasks import Task, elevation_task, roads_task, treecover_task, population_task, nightlights_task, income_task
from dataset.dataloader import EmbeddedDataset

def plot_tsne(dataset, filename):
    # perplexity values tried: 5, 40, 25
    # you can tune values for tsne
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300, init="pca")
    X = dataset.X
    y = dataset.y 

   # create new Pandas dataframe
    df = pd.DataFrame()
    df['y'] = y

#    sc = StandardScaler()
#    sc.fit(X)
#    X_std = sc.transform(X)

    tsne_results = tsne.fit_transform(X)

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    # get the range of the values in df['y']
    y_range = df['y'].max() - df['y'].min()
    print("Y RANGE ", y_range)

    bins = np.array([0, int(0.25 * y_range), int(0.5 * y_range), int(0.75 * y_range), y_range])
    print("BINS ", bins)

    inds = np.digitize(df['y'], bins)
    df['y'] = inds

    y_range = df['y'].max() - df['y'].min()
    print("Y RANGE ", y_range)

    plt.clf()

    sns_plot = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 5),
        data=df,
        legend="full",
        alpha=0.1
    )

    sns_plot.figure.savefig("plots/tsne/tsne_alltaskrun/" + filename + ".png")

if __name__ == "__main__":
    # load embeddings 
    model_name = "pretrained_visiontransformer_1024_ElRdInTr"

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

    embedded_dataset_nightlights = EmbeddedDataset(embeddings, nightlights_task)
    embedded_dataset_mosaiks_nightlights = EmbeddedDataset(mosaiks_embeddings, nightlights_task)

    # create new embedded_dataset_income
    embedded_dataset_income = EmbeddedDataset(embeddings, income_task)
    # create new embedded_dataset_mosaiks_income
    embedded_dataset_mosaiks_income = EmbeddedDataset(mosaiks_embeddings, income_task)

    # create new embedded_dataset_population
    embedded_dataset_population = EmbeddedDataset(embeddings, population_task)
    # create new embedded_dataset_mosaiks_population
    embedded_dataset_mosaiks_population = EmbeddedDataset(mosaiks_embeddings, population_task)

    # create new embedded_dataset_roads
    embedded_dataset_roads = EmbeddedDataset(embeddings, roads_task)
    # create new embedded_dataset_mosaiks_roads
    embedded_dataset_mosaiks_roads = EmbeddedDataset(mosaiks_embeddings, roads_task)

    plot_tsne(embedded_dataset_elevation, "tsne_elevation_perp30_pca_normalized_scaled")
    plot_tsne(embedded_dataset_mosaiks_elevation, "tsne_mosaiks_elevation")
    plot_tsne(embedded_dataset_treecover, "tsne_treecover_perp30")
    plot_tsne(embedded_dataset_mosaiks_treecover, "tsne_mosaiks_treecover_perp30")
    plot_tsne(embedded_dataset_mosaiks_nightlights, "tsne_mosaiks_nightlights")
    plot_tsne(embedded_dataset_nightlights, "tsne_nightlights")
    plot_tsne(embedded_dataset_mosaiks_income, "tsne_mosaiks_income")
    plot_tsne(embedded_dataset_income, "tsne_income")
    plot_tsne(embedded_dataset_mosaiks_population, "tsne_mosaiks_population")
    plot_tsne(embedded_dataset_population, "tsne_population")
    plot_tsne(embedded_dataset_mosaiks_roads, "tsne_mosaiks_roads")
    plot_tsne(embedded_dataset_roads, "tsne_roads")

# Plot already generated: stored in plots/tsne_elevation.png
#    plot_tsne(embedded_dataset_elevation, "embeddings elevation")

#    plot_tsne(embedded_dataset_mosaiks_elevation, "mosaiks_embeddings")