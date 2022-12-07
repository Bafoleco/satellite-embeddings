import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopy.distance

# setting path
sys.path.append('../..')


import util.embedding_utils as embedding_utils
import util.util as util

def create_lat_lon_map():
    # load grid
    file_name = '../../../data/int/grids/CONTUS_16_640_UAR_100000_0.npz'
    grid = np.load(file_name)

    lat = grid['lat']
    lon = grid['lon']
    ids = grid['ID']

    string_sep = '_'

    data = zip(ids, lat, lon)
    df = pd.DataFrame(data, columns = ['ids', 'lat', 'lon'])

    return df

def plot_distance_comp(distances, embedded_distances, name):


    plt.scatter(distances, embedded_distances, label="Ours")


    # plt.scatter(distances, embedded_distance_mosaiks, label="MOSAIKS")
    plt.xlabel("Real Distance")
    plt.ylabel("Embedded Distance")

    # add line of best fit
    z = np.polyfit(distances, embedded_distances, 1)
    p = np.poly1d(z)
    
    plt.plot(distances, p(distances), "r--")

    plt.legend()

    # save the plot
    plt.savefig("./embedding_experiments/plots/" + name + ".png")

    # clear the plot
    plt.clf()

if __name__ == "__main__":
    # load embeddings 
    model_name = "pretrained_resnet"

    with open('../../embeddings/' + util.get_embedding_filename(model_name), 'rb') as f:
        embeddings = pickle.load(f)

    with open('../../../data/int/CONTUS_UAR.pkl', 'rb') as f:
        mosaiks_embeddings = pickle.load(f)
        X = mosaiks_embeddings["X"]
        ids_X = mosaiks_embeddings["ids_X"]
        mosaiks_embeddings = embedding_utils.mosaiks_format_to_map(X, ids_X, embeddings)


    lat_lon_map = create_lat_lon_map()

    print("Lat Lon Map: ", lat_lon_map)


    distances = []
    embedded_distance_mosaiks = []
    embedded_distance = []

    # pick 1000 random pairs
    for i in range(5000):
        # report progress
        if i % 100 == 0:
            print("Progress: ", i)

        # pick two random keys
        key1 = np.random.choice(list(embeddings.keys()))
        key2 = np.random.choice(list(embeddings.keys()))

        # get the embeddings
        emb1 = embeddings[key1]
        emb2 = embeddings[key2]

        # get the mosaiks embeddings
        mosaiks_emb1 = mosaiks_embeddings[key1]
        mosaiks_emb2 = mosaiks_embeddings[key2]

        # compute the real distance

        lat1 = lat_lon_map[lat_lon_map['ids'] == key1]['lat'].values[0]
        lon1 = lat_lon_map[lat_lon_map['ids'] == key1]['lon'].values[0]

        lat2 = lat_lon_map[lat_lon_map['ids'] == key2]['lat'].values[0]
        lon2 = lat_lon_map[lat_lon_map['ids'] == key2]['lon'].values[0]

        coords_1 = (lat1, lon1)
        coords_2 = (lat2, lon2)

        real_distance = geopy.distance.distance(coords_1, coords_2).km

        distances.append(real_distance)

        # compute the embedded distance
        embedded_distance.append(np.linalg.norm(emb1 - emb2))

        # print(np.linalg.norm(emb1 - emb2))

        # compute the mosaiks embedded distance
        embedded_distance_mosaiks.append(np.linalg.norm(mosaiks_emb1 - mosaiks_emb2))


    # get mean distances
    mean_real_distance = np.mean(distances)
    mean_embedded_distance = np.mean(embedded_distance)
    mean_embedded_distance_mosaiks = np.mean(embedded_distance_mosaiks)

    # divide by mean
    distances = [x / mean_real_distance for x in distances]
    embedded_distance = [x / mean_embedded_distance for x in embedded_distance]
    embedded_distance_mosaiks = [x / mean_embedded_distance_mosaiks for x in embedded_distance_mosaiks]

    # plot our data
    plot_distance_comp(distances, embedded_distance, "ours")

    # plot mosaiks data
    plot_distance_comp(distances, embedded_distance_mosaiks, "mosaiks")

    # compute the correlation
    print("Correlation between real and embedded distance: ", np.corrcoef(distances, embedded_distance)[0, 1])

    # compute the correlation 
    print("Correlation between real and mosaiks embedded distance: ", np.corrcoef(distances, embedded_distance_mosaiks)[0, 1])

