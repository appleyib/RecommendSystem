from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd


def combinedIdandtags(movies_frame, tags_frame):
    joined_table = pd.merge(movies_frame, tags_frame, on='movieId', how='left')
    joined_table.fillna("", inplace=True)
    joined_table = pd.DataFrame(joined_table.groupby('movieId')['tag'].apply(
        lambda x: "%s" % ' '.join(x)))

    return joined_table

def getTF(meta):
    t_feature = TfidfVectorizer(stop_words='english')
    t_matrix = t_feature.fit_transform(meta['metadata'])
    t_feature = pd.DataFrame(t_matrix.toarray(), index=meta.index.tolist())

    return t_feature

def createmetaforTF(movies_frame, joined_frame):
    meta = pd.merge(movies_frame, joined_frame, on='movieId', how='left')
    meta['metadata'] = meta[['tag', 'genres']].apply(
        lambda x: ' '.join(x), axis=1)

    return meta

def model_training(movies_frame, tf, meta,filtered_ratings, n=100):
    svd = TruncatedSVD(n_components=n)
    content_1 = svd.fit_transform(tf)
    content = pd.DataFrame(content_1, index=meta.title.tolist())
    ratings_f1 = pd.merge(movies_frame['movieId'], filtered_ratings, on="movieId", how="right")
    ratings_f2 = ratings_f1.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    svd = TruncatedSVD(n_components=n)
    collaborative_1 = svd.fit_transform(ratings_f2)
    collaborative = pd.DataFrame(collaborative_1, index=meta.title.tolist())

    print("finish training.. start saving")

    content.to_pickle('./Files/latent_content.pkl')
    collaborative.to_pickle('./Files/latent_collaborative.pkl')

    print("saving finished")


# training process
movies = pd.read_csv('./data/movies.csv')
tags = pd.read_csv('./data/tags_shuffled_rehashed.csv')
ratings = pd.read_csv('./data/train_ratings_binary.csv')
#preprocessing because the matrix is too big

filtered_ratings = ratings.groupby('userId').filter(lambda x: len(x) >= 50)
movies = movies[movies.movieId.isin(filtered_ratings.movieId.unique().tolist())]

# map movie to id: because we have filtered some of ids

movies2id_dict = dict(zip(movies.title.tolist(), movies.movieId.tolist()))
movies_tags = combinedIdandtags(movies, tags)
meta = createmetaforTF(movies, movies_tags)
tf = getTF(meta)
model_training(movies, tf, meta, filtered_ratings, 200)
filtered_ratings.to_pickle('./Files/rating.pkl')

