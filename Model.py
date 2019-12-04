import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

collabSim = np.load("collabSim.npy")
contentSim = np.load("contentSim.npy")
nLarge = np.load("nLarge.npy")


class ContentBased(object):

    def __init__(self, latent_content):
        """top N memory based: content
        """
        self.content = latent_content

    def predict_top_n(self, query, n=10):
        a_1 = np.array(self.content.loc[query]).reshape(1, -1)
        content = cosine_similarity(self.content, a_1).reshape(-1)
        dictDf = {'content': content}
        similar = pd.DataFrame(dictDf, index=self.content.index)
        similar.sort_values('content', ascending=False, inplace=True)
        return similar.head(n+1)[1:].index.tolist()


class CollabBased(object):

    def __init__(self, latent_collab):
        """top N memory based: collaborative
        """
        self.collab = latent_collab

    def predict_top_n(self, query, n=10):

        a_1 = np.array(self.collab.loc[query]).reshape(1, -1)
        collab = cosine_similarity(self.collab, a_1).reshape(-1)
        dictDf = {'collaborative': collab}
        similar = pd.DataFrame(dictDf, index=self.collab.index)
        similar.sort_values('collaborative', ascending=False, inplace=True)
        return similar.head(n+1)[1:].index.tolist()


class HybridBased(object):

    def __init__(self, latent_content, latent_collab):
        """top N memory based: hybrid
        """
        self.content = latent_content
        self.collab = latent_collab

    def predict_top_n(self, query, n=10):
        a_1 = np.array(self.content.loc[query]).reshape(1, -1)
        a_2 = np.array(self.collab.loc[query]).reshape(1, -1)
        content = cosine_similarity(self.content, a_1).reshape(-1)
        collaborative = cosine_similarity(self.collab, a_2).reshape(-1)
        curContentSim = contentSim[self.content.loc(query)]
        curCollabSim = collabSim[self.collab.loc(query)]
        hybrid = ((content + collaborative)/2.0)
        # a data frame of movies based on similarity to query
        dictDf = {'hybrid': hybrid}
        similar = pd.DataFrame(dictDf, index=self.content.index)
        # similar.sort_values('hybrid', ascending=False, inplace=True)
        # sorting is too in efficiency, we can use pandas.Dataframe.nsmallest to find n most simillar dataframe
        similar = similar.nsmallest(n+1, 'hybrid', keep='all')
        return similar.head(n+1)[1:].index.tolist()