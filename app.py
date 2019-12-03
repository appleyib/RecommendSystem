
import pickle
import numpy as np
from Model import ContentBased, CollabBased, HybridBased, ModelBased
import pandas as pd
import csv
import collections

# the recommender module
class Recommender:

    def __init__(self):
        with open('./Files/map.pkl', 'rb') as f:
            self.movie_map = pickle.load(f)
        with open('./Files/rating.pkl', 'rb') as f:
            self.rating = pickle.load(f)
        with open('./Files/latent_collaborative.pkl', 'rb') as f:
            latent_collab = pickle.load(f)
        with open('./Files/latent_content.pkl', 'rb') as f:
            latent_content = pickle.load(f)

        self.imap = dict(zip(self.movie_map.values(), self.movie_map.keys()))
        self.clf_content = ContentBased(latent_content)
        self.clf_collab = CollabBased(latent_collab)
        self.clf_hybrid = HybridBased(latent_content, latent_collab)

    def parsing_args(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('movie', required=False,
                                 help="movie title followed by year")
        self.parser.add_argument('limit', required=False,
                                 help="N in top N films")

    def get_all_recommendations(self, moviename, n):
        if moviename in self.movie_map.keys():

            output = {
                'content': {'content':
                            self.clf_content.predict_top_n(moviename, n)},
                'collaborative': {'collaborative':
                                  self.clf_collab.predict_top_n(moviename, n)},
                'hybrid': {'hybrid':
                           self.clf_hybrid.predict_top_n(moviename, n)},
                     }
        else:
            output = None
        return output

    def get_all_recommendationsbyid(self, id, n):
        if id in self.imap.keys():

            moviename = self.imap[id]

            output = {
                'content': {'content':
                            self.clf_content.predict_top_n(moviename, n)},
                'collaborative': {'collaborative':
                                  self.clf_collab.predict_top_n(moviename, n)},
                'hybrid': {'hybrid':
                           self.clf_hybrid.predict_top_n(moviename, n)},
                     }
        else:
            output = None
        return output


'''ratings = pd.read_csv('./data/train_ratings_binary.csv')
voting = collections.defaultdict(set)
for i in range(len(ratings)):
    uid = ratings.iloc[i]['userId']
    movieid = ratings.iloc[i]['movieId']
    label = ratings.iloc[i]['rating']
    if label == 1:
        voting[uid].add(movieid)
        print(uid,movieid)
with open('dict.pickle', 'wb') as handle:
    pickle.dump(voting, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("writing complete")'''
with open('dict.pickle', 'rb') as handle:
    voting = pickle.load(handle)

rows = []
with open("data/result.csv","w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Id","rating"])

    r = Recommender()
    movie_map = r.movie_map
    testset = pd.read_csv('./data/test_ratings.csv')
    for i in range(2299491, len(testset)):
        uid = testset.iloc[i]['userId']
        movieid = testset.iloc[i]['movieId']
        try:
            res = r.get_all_recommendationsbyid(movieid, 10)
        except ValueError:
            res = None
        pred = 0.0
        if res is not None:
            for name in res['hybrid']['hybrid']:
                if movie_map[name] in voting[uid]:
                    pred = 1.0

        rows.append([i, pred])
        if i%10000 == 0 or i == len(testset)-1:
            print("we have write ", i)
            writer.writerows(rows)
            rows=[]









#print(ratings.head(3))
#r= Recommender()
#res = r.get_all_recommendations("Jumanji (1995)", 5)
#print(res)


