import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

movies = pd.read_csv('./data/movies.csv')
tags = pd.read_csv('./data/tags_shuffled_rehashed.csv')
ratings = pd.read_csv('./data/train_ratings_binary.csv')

