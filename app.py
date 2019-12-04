
import pickle
import numpy as np
from Model import ContentBased, CollabBased, HybridBased, ModelBased
import pandas as pd
import csv

def movieid2matrixid(movieid, movieid2name, name2matrixid):
    if movieid in movieid2name:
        return name2matrixid[movieid2name[movieid]]
    else:
        return None


with open('./Files/map.pkl', 'rb') as f:
    movie_map = pickle.load(f)


imap = dict(zip(movie_map.values(), movie_map.keys()))

matrixid2moviename = dict()
name2matrixid = dict()
with open('./Files/latent_content.pkl', 'rb') as f:
    latent_content = pickle.load(f)

nLarge = np.load("nLarge.npy")

for i in range(len(nLarge)):
    matrixid2moviename[i] = latent_content.index[i]
    name2matrixid[latent_content.index[i]] = i



with open('dict.pickle', 'rb') as handle:
    voting = pickle.load(handle)

rows = []
with open("data/result.csv","w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Id","rating"])

    movieid2name = imap
    testset = pd.read_csv('./data/test_ratings.csv')
    for i in range(2299491, len(testset)):
        uid = testset.iloc[i]['userId']
        movieid = testset.iloc[i]['movieId']
        matrixid = movieid2matrixid(movieid, movieid2name, name2matrixid)

        pred = 0.0
        if matrixid is not None:
            for mid in nLarge[matrixid]:
                name = matrixid2moviename[mid]
                if name in voting[uid]:
                    pred = 1.0
                    

        rows.append([i, pred])
        if i%10000 == 0 or i == len(testset)-1:
            print("we have write ", i)
            writer.writerows(rows)
            rows=[]