import numpy as np
import pandas as pd
import sys

l = len(sys.argv)
collab_feat_num = None
cont_feat_num = None

# parses arguments
for i in range(1,len(sys.argv)):
    if sys.argv[i] == "-cfn":
        collab_feat_num = int(sys.argv[i+1])
    elif sys.argv[i] == "-contn":
        cont_feat_num = int(sys.argv[i+1])

# movie nums
movie_num = 27278
tags_num = 1128
user_num = 138493
#user_num = 1000

# loads files
print("Loading files...")
# movies
with open('data/movies.csv', encoding='gb18030',errors='ignore') as csvfile:
    df_movie = pd.read_csv(csvfile)
    mov_id = df_movie['movieId'].to_numpy(dtype='int')
    all_tag = df_movie['genres'].tolist()
    id_row_num_map = dict(zip(df_movie['movieId'].tolist(), list(range(movie_num))))

# genres
with open('data/genome-tags.csv') as csvfile:
    df = pd.read_csv(csvfile)
    tag_id_map = dict(zip(df['tag'].tolist(), df['tagId'].tolist()))

# movie feature matrix
mov_fea = np.zeros((movie_num,tags_num))

# adds genres to movies without tags
for i in range(movie_num):
    tag_list = all_tag[i].split('|')
    for j in [tag_id_map[tag.lower()]-1 for tag in tag_list if tag.lower() in tag_id_map]:
        mov_fea[i, j] = 1

# genome scores
with open('data/genome-scores.csv') as csvfile:
    df = pd.read_csv(csvfile)
    mov_rel = df['relevance'].to_numpy(dtype='float').reshape((10381,1,tags_num)) 
    mov_id_with_tags = df['movieId'].unique()
    
mov_fea[np.argwhere(np.isin(mov_id, mov_id_with_tags))] = mov_rel

# movies ratings
with open('data/train_ratings_binary.csv') as csvfile:
    df_train = pd.read_csv(csvfile)

with open('data/val_ratings_binary.csv') as csvfile:
    df_val = pd.read_csv(csvfile)   
    
df_train = pd.concat([df_train, df_val], ignore_index=True)

# test data
with open('data/test_ratings.csv') as csvfile:
#with open('data/val_ratings_binary.csv') as csvfile:
    df_test = pd.read_csv(csvfile)  

print("Loading file complete, now generating/assembling feats...")