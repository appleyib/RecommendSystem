import numpy as np
import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def superHybrid(mov_fea, mov_id, df_train, df_test, df_val, user_num=1000):
	df_train = df_train[df_train["userId"]<=user_num]

	# to calculate temp accuracy!
	df_test = df_val

	df_test = df_test[df_test["userId"]<=user_num]
	df_movie = pd.DataFrame(mov_fea, index = mov_id)
	df_user_movie_rat = pd.merge(df_train, df_movie, left_on="movieId", right_index=True, how="left")
	df_train = df_user_movie_rat.copy()
	df_user_movie_rat[df_user_movie_rat["rating"] == 0]= -df_user_movie_rat
	df_user_movie_rat.drop(["movieId", "rating"], axis=1, inplace=True)
	df_user_movie_rat["userId"]= abs(df_user_movie_rat)
	df_user_movie_rat = df_user_movie_rat.groupby("userId").mean()
	df_user_movie_rat.columns = range(2000, 2000+mov_fea.shape[1])
	# print(df_train.columns)
	# print(df_user_movie_rat.columns)
	df_train = pd.merge(df_train, df_user_movie_rat, left_on="userId", right_index=True, how="left")
	y = df_train["rating"]
	print(df_train)
	df_train.drop(["rating", "movieId"], axis = 1, inplace=True)
	print("columns of training datasets:")
	print(df_train.columns)
	print("training...")
	clf = RandomForestClassifier(max_depth=10, n_estimators=100)
	clf.fit(df_train, y)

	# df_test['movieId'] = df_test['movieId'].map(id_row_num_map)
	# df_test_array = df_test.to_numpy(dtype='int')
	# user_test = df_test.groupby('userId')
	# final = np.zeros((0,))
	print("training done, predicting...")

	print(df_test.columns)
	df_test=pd.merge(df_test, df_movie, left_on="movieId", right_index=True, how="left")
	df_test=pd.merge(df_test, df_user_movie_rat, on="userId", how="left")
	truth = df_test["rating"]
	df_test.drop(["movieId", "rating"], axis=1, inplace=True)
	print("columns of test datasets:")
	print(df_test.columns)
	print("predicting...")
	predict = clf.predict(df_test)
	print(accuracy_score(truth, predict))

	