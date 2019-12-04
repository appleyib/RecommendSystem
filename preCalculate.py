import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

index = None

# calculates the cosine similarity of between each pair of data
def pre_calculate_cosine_sim(fileP):
	data = None
	with open(fileP, 'rb') as f:
		data = pickle.load(f)
	if data is not None:
		data_sim = cosine_similarity(data, data)
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
collab_cosine_sim = None
content_cosine_sim = None

# arguments defines whether we use collaborative feature or we use content feature or both
# uses collab feat
if len(sys.argv) >= i+2 and sys.argv[i] == "-collab":
	collab_cosine_sim = pre_calculate_cosine_sim(sys.argv[i+1])
	# outFile = open("collabSim.npy", "wb")
	# np.save(outFile, collab_cosine_sim)
	i += 2
	print("Collab consine cosine_similarity calculation complete!")

# uses content feat
if len(sys.argv) >= i+2 and sys.argv[i] == "-content":
	content_cosine_sim = pre_calculate_cosine_sim(sys.argv[i+1])
	# outFile = open("contentSim.npy", "wb")
	# np.save(outFile, content_cosine_sim)
	i += 2
	print("Content consine cosine_similarity calculation complete!")

# the variable to store the final similarity
sim = None
if collab_cosine_sim is None:
	sim = content_cosine_sim
elif content_cosine_sim is None:
	sim = collab_cosine_sim
else:
	sim = content_cosine_sim + collab_cosine_sim

nLarge = pre_calculate_n_largest(sim, 10)
outFile = open("nLarge.npy", "wb")
np.save(outFile, nLarge)
print("N largest sim for each movie calculation complete!")
print("Pre calculation done!")

