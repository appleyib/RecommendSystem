import numpy as np
import pandas as pd
import sys
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from assembleFeats import assembleFeats, reduceContentFeats
from superHybrid import superHybrid

l = len(sys.argv)
collab_feat_num = None
cont_feat_num = None
superHybridFlag = False

# parses arguments
for i in range(1,len(sys.argv)):
    if sys.argv[i] == "-cfn":
        collab_feat_num = int(sys.argv[i+1])
    elif sys.argv[i] == "-contn":
        cont_feat_num = int(sys.argv[i+1])
    elif sys.argv[i] == "-superHybrid":
        superHybridFlag = True


# movie nums
movie_num = 27278
tags_num = 1128
user_num = 138493
# user_num = 1000

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

if superHybridFlag is True:
    print("Loading file complete. Running Yizhao Wang's super stupid hybrid model...")
    superHybrid(mov_fea, mov_id, df_train, df_test, df_val, user_num)
    exit(0)

print("Loading file complete, now generating/assembling feats...")


# uses SVD tu reduce the dimensionality of content feats (mov_fea)
if cont_feat_num is None:
    print("Running in non-SVD reduction mode, content feats dimensionality will not be reduced.")
else:
    mov_fea = reduceContentFeats(mov_fea, cont_feat_num)
    print("SVD/PCA on content feats done!")

# adds collaborative features decomposed by PCA/SVD

if collab_feat_num is None:
    print("Running in non-collaborative filtering mode, collab feats will not be assembled.")
else:
    mov_fea = assembleFeats(df_movie, df_train, mov_fea, collab_feat_num)
    print("Assembling feats done!")


df_train['movieId'] = df_train['movieId'].map(id_row_num_map)
df_test['movieId'] = df_test['movieId'].map(id_row_num_map)

user_train = df_train.groupby('userId')
df_train_array = df_train.to_numpy(dtype='int')
user_test = df_test.groupby('userId')
df_test_array = df_test.to_numpy(dtype='int')


# starts to training a classifier for each user
# will predict for samples in the test case as well since
# we want to save memory
final = np.zeros((0,))
for n in range(1,user_num+1):
    
    train = df_train_array[user_train.groups[n]]   
    test = df_test_array[user_test.groups[n]]   

    
    X = mov_fea[train[:,1]]
    y = train[:,2]
    
    #clf = KNeighborsClassifier(n_neighbors=10)
    #clf = tree.DecisionTreeClassifier(max_depth=10)
    clf = RandomForestClassifier(max_depth=4, n_estimators=200)
    clf.fit(X, y) 
    

    Z = mov_fea[test[:,1]]
    final = np.concatenate((final,clf.predict(Z))) 
    if n%100 ==0:
        print(n)
 
#truth = df_test.to_numpy(dtype='float')[0:final.shape[0],2]
#fpr, tpr, thresholds = metrics.roc_curve(truth, final)
#print(metrics.auc(fpr, tpr))  

with open('data/kaggle_sample_submission.csv') as csvfile:
    df_res = pd.read_csv(csvfile)
    df_res['rating'] = pd.Series(final)
    
df_res.to_csv('data/kaggle_sample_submission.csv', index=False)
