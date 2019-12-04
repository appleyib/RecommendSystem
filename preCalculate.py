import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance

index = None

# calculates the cosine similarity between each pair of data
def pre_calculate_cosine_sim(fileP):
	data = None
	with open(fileP, 'rb') as f:
		data = pickle.load(f)
	print("data size:", data.shape)
	if data is not None:
		data_sim = cosine_similarity(data, data)
	else:
		print("No collaborative pkl file!")
	return data_sim

# calculates the squared error similarity between each pair of data
def pre_calculate_se_sim(fileP):
	data = None
	with open(fileP, 'rb') as f:
		data = pickle.load(f)
	print("data size:", data.shape)
	if data is not None:
		data_sim = -euclidean_distances(data, data)
	else:
		print("No collaborative pkl file!")
	return data_sim

# calculates the n movies with largest similarity for each movie (except its own)
def pre_calculate_n_largest(data, n):
	l = data.shape[0]
	nLarge = np.array([(-data[i]).argpartition(n)[:n] for i in range(l)])
	print("Shape of the n most similar movies to each movie is:", nLarge.shape)
	print(nLarge)
	return nLarge

# sets up variables
i=1
collab_sim = None
content_sim = None
# flag whether we should use squared error distance
SE_distance = False

if len(sys.argv) >= i+1 and sys.argv[i] == "-SE":
	SE_distance = True
	i += 1

# arguments defines whether we use collaborative feature or we use content feature or both
# uses collab feat
if len(sys.argv) >= i+2 and sys.argv[i] == "-collab":
	if SE_distance:
		collab_sim = pre_calculate_se_sim(sys.argv[i+1])
	else:
		collab_sim = pre_calculate_cosine_sim(sys.argv[i+1])
	# outFile = open("collabSim.npy", "wb")
	# np.save(outFile, collab_cosine_sim)
	i += 2
	print("Collab similarity calculation complete!")

# uses content feat
if len(sys.argv) >= i+2 and sys.argv[i] == "-content":
	if SE_distance:
		content_sim = pre_calculate_se_sim(sys.argv[i+1])
	else:
		content_sim = pre_calculate_cosine_sim(sys.argv[i+1])
	# outFile = open("contentSim.npy", "wb")
	# np.save(outFile, content_cosine_sim)
	i += 2
	print("Content similarity calculation complete!")


# the variable to store the final similarity
sim = None
if collab_sim is None:
	sim = content_sim
elif content_sim is None:
	sim = collab_sim
else:
	sim = content_sim + collab_sim * 0.2

nLarge = pre_calculate_n_largest(sim, 10)
outFile = open("nLarge.npy", "wb")
np.save(outFile, nLarge)
print("N largest sim for each movie calculation complete!")
print("Pre calculation done!")

