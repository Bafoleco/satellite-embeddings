import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns

import pickle
import matplotlib.pyplot as plt

sys.path.append('../..')

import util.embedding_utils as embedding_utils
import util.util as util
import dataset.dataloader as dataloader
from dataset.tasks import Task, elevation_task, roads_task
from dataset.dataloader import EmbeddedDataset

def plot_pca(dataset, name):
    X = dataset.X
    y = dataset.y 

    sc = StandardScaler()
    sc.fit(X)
    X_std = sc.transform(X)

    pca = PCA(n_components=4)
    pc = pca.fit_transform(X_std)

    # print explained variance ratio
    print("Explained variance ratio: ", pca.explained_variance_ratio_)

    # plot explained variance ratio

    print("PC ", pc)

    pc_df = pd.DataFrame(data=pc, columns = ['PC1', 'PC2','PC3','PC4'])
    pc_df['Cluster'] = y
    pc_df.head()

    # plot a barplot of each component and its explained variance ratio
#    plt.plot(np.cumsum(pca.explained_variance_ratio_))

    df = pd.DataFrame({'var': pca.explained_variance_ratio_,
            'PC':['PC1','PC2','PC3','PC4']})

  #  sns.barplot(x='PC',y="var", 
   #        data=df, color="c")

    print("Plotting PCA")
    
    # TODO: Try a scatter plot method with seaborn, similar to t-SNE!!

    bins = np.array([0, 1000, 2000, 3000, 4000])
    inds = np.digitize(df['y'], bins)
    df['y'] = inds

    y_range = df['y'].max() - df['y'].min()
    print("Y RANGE ", y_range)

    sns_plot = sns.scatterplot(
        x=df['PC1'], y=df['PC2'],
        hue="y",
        palette=sns.color_palette("hls", 5),
        data=df,
        legend="full",
        alpha=0.3
    )

    # NOTE: This takes way too long!!
    sns_plot = sns.lmplot(x="PC1", y="PC2",
        data=pc_df, 
        fit_reg=False, 
        hue='Cluster', # color by cluster
        legend=True,
        ci=None,
        scatter_kws={"s": 80}) # specify the point size

    sns_plot.figure.savefig("./plots/pca_" + name + ".png")

if __name__ == "__main__":
    # load embeddings 
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

    plot_pca(embedded_dataset_elevation, "embeddings_elevation")