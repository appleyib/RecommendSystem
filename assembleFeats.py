import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

latent_feats_file_name = "latent_feats.npy"
assembled_feats_file_name = "assembled_feats.npy"

# the function to assemble extended movies features by adding collab feats generating from SVD/PCA
def assembleFeats(df, df_train, mov_fea, collab_feat_num):
	df_train = df_train.groupby("userId").filter(lambda x: len(x) >= 55)
	movie_num = mov_fea.shape[0]
	tags_num = mov_fea.shape[1]
	collabFeats = np.zeros((movie_num, collab_feat_num))

	# gets table with colmn userid and row movid with values ratings (1 vs -1)
	all_collab_filter = df_train.pivot(index='movieId', columns='userId',values='rating')
	all_collab_filter[all_collab_filter==0] = -1
	all_collab_filter = all_collab_filter.fillna(0)
	# all_collab_filter_include_non_rated = pd.merge(df['movieId'], all_collab_filter, on='movieId', how='left').fillna(0)

	# defines svd classifier
	svd = TruncatedSVD(n_components=collab_feat_num)
	# use svd to decompose collab feats
	latent_collab_feats = pd.DataFrame(svd.fit_transform(all_collab_filter), index=all_collab_filter.index.tolist())
	all_latent_feats_include_non_rated = pd.merge(df['movieId'], latent_collab_feats, left_on='movieId', right_index=True, how='left').fillna(0).to_numpy(dtype='float')
	print("Tag feats shape is:", mov_fea.shape)
	print("Latent collab feats shape is:", all_latent_feats_include_non_rated.shape)
	all_latent_feats_include_non_rated.save(latent_feats_file_name)
	# assembles!
	assembledFeats = np.concatenate((mov_fea, latent_collab_feats), axis = 1)
	assembledFeats.save(assembled_feats_file_name)
	print("Assembled feats shape is:", assembledFeats.shape)

	return assembledFeats

