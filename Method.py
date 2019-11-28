import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import sys

movies = pd.read_csv('./data/movies.csv')
tags = pd.read_csv('./data/tags_shuffled_rehashed.csv')
ratings = pd.read_csv('./data/train_ratings_binary.csv')

# limit ratings to user ratings that have rated more that 55 movies
# it also filters the number of movies we can keep-- the reason is my
# laptop limited power.
ratings_f = ratings.groupby('userId').filter(lambda x: len(x) >= 55)
movie_list_rating = ratings_f.movieId.unique().tolist()
# filter the movies data frame
movies = movies[movies.movieId.isin(movie_list_rating)]
# map movie to id:
Mapping_file = dict(zip(movies.title.tolist(), movies.movieId.tolist()))
# remove unnecessary timesteps
# make a useful dataframe from tags and movies
mixed = pd.merge(movies, tags, on='movieId', how='left')

# create metadata from all tags and genres
mixed.fillna("", inplace=True)
mixed = pd.DataFrame(mixed.groupby('movieId')['tag'].apply(
    lambda x: "%s" % ' '.join(x)))
Final = pd.merge(movies, mixed, on='movieId', how='left')
Final['metadata'] = Final[['tag', 'genres']].apply(
    lambda x: ' '.join(x), axis=1)

# text transformation and truncated SVD to create a content latent matrix:
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(Final['metadata'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=Final.index.tolist())
svd = TruncatedSVD(n_components=200)
latent_matrix_1 = svd.fit_transform(tfidf_df)
latent_matrix_1_df = pd.DataFrame(
                        latent_matrix_1,
                        index=Final.title.tolist())

# text transformation and truncated SVD to create a collaborative
# latent matrix:
ratings_f1 = pd.merge(movies['movieId'], ratings_f,
                        on="movieId", how="right")
ratings_f2 = ratings_f1.pivot(
                        index='movieId', columns='userId',
                        values='rating').fillna(0)
svd = TruncatedSVD(n_components=200)
latent_matrix_2 = svd.fit_transform(ratings_f2)
latent_matrix_2_df = pd.DataFrame(
                        latent_matrix_2,
                        index=Final.title.tolist())




# pickle all necessary files in ./Files/:
ratings_f.to_pickle('./Files/rating.pkl')
ratings_f2.to_pickle('./Files/rating_f2.pkl')
latent_matrix_1_df.to_pickle('./Files/latent_content.pkl')
latent_matrix_2_df.to_pickle('./Files/latent_collaborative.pkl')
with open('./Files/map.pkl', 'wb') as f:
     pickle.dump(Mapping_file, f, pickle.HIGHEST_PROTOCOL)
