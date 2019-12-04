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
from assembleFeats import assembleFeats

l = len(sys.argv)
collab_feat_num = None

# parses arguments
for i in range(1,len(sys.argv)):
    if sys.argv[i] == "-cfn":
        collab_feat_num = int(sys.argv[i+1])

# movie nums
movie_num = 27278
tags_num = 1128
user_num = 138493

# loads files
print("Loading files...")
# movies
with open('data/movies.csv', encoding='gb18030',errors='ignore') as csvfile:
    df = pd.read_csv(csvfile)
    mov_id = df['movieId'].to_numpy(dtype='int')

mov_fea = np.zeros((movie_num,tags_num))

# genome scores
with open('data/genome-scores.csv') as csvfile:
    df = pd.read_csv(csvfile)
    mov_rel = df['relevance'].to_numpy(dtype='float').reshape((10381,1,tags_num)) 
    mov_id_with_tags = df['movieId'].unique()
    
mov_fea[np.argwhere(np.isin(mov_id, mov_id_with_tags))] = mov_rel

# movies ratings
with open('data/train_ratings_binary.csv') as csvfile:
    df_train = pd.read_csv(csvfile)

# test data
with open('data/test_ratings.csv') as csvfile:
    df_val = pd.read_csv(csvfile)  

# adds collaborative features decomposed by PCA/SVD
print("Loading file complete, now generating/assembling feats...")
if collab_feat_num is None:
    print("Running in non-collaborative filtering mode, collab feats will not be assembled.")
else:
    mov_fea = assembleFeats(df, df_train, mov_fea, collab_feat_num)

# starts to training a classifier for each user
# will predict for samples in the test case as well since
# we want to save memory
final = np.zeros((0,))
for n in range(1,user_num+1):
    user_train = df_train[df_train['userId']==n].to_numpy(dtype='int')   
    user_val = df_val[df_val['userId']==n].to_numpy(dtype='int')   

    
    X = np.squeeze(mov_fea[np.argwhere(np.isin(mov_id, user_train[:,1]))])
    y = user_train[:,2]
    
    #clf = KNeighborsClassifier(n_neighbors=10)
    #clf = tree.DecisionTreeClassifier(max_depth=5)
    clf = RandomForestClassifier(max_depth=2, n_estimators=20)
    clf.fit(X, y) 
    

    Z = np.squeeze(mov_fea[np.argwhere(np.isin(mov_id, user_val[:,1]))])
    final = np.concatenate((final,clf.predict(Z))) 
    if n%100 ==0:
        print(n)
 
#truth = df_val.to_numpy(dtype='float')[0:final.shape[0],2]
#fpr, tpr, thresholds = metrics.roc_curve(truth, final)
#print(metrics.auc(fpr, tpr))  

with open('data/kaggle_sample_submission.csv') as csvfile:
    df_res = pd.read_csv(csvfile)
    df_res['rating'] = pd.Series(final)
    
df_res.to_csv('data/kaggle_sample_submission.csv', index=False)