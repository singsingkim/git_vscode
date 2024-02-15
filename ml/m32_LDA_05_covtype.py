from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

datasets = fetch_covtype()
x = datasets.data  
y = datasets.target

# print(x.shape,y.shape)      #(581012, 54) (581012,)
# print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747

# y = pd.get_dummies(y)
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y.reshape(-1,1))

# print(y,y.shape,sep='\n')
# print(np.count_nonzero(y[:,0]))
'''
sklearn : (581012, 7)
pandas  : (581012, 7)
keras   : (581012, 8)
keras 첫번째 열이 미심직어 찍어보니
print(np.count_nonzero(y[:,0])) # 0
따라서 첫번째 열 잘라내고 슬라이싱
'''
# print(y.shape)

# print(y,y.shape,sep='\n')       # (581012, 7)
# print(np.count_nonzero(y[:,0])) # 211840

x = StandardScaler().fit_transform(x)
lda = LinearDiscriminantAnalysis().fit(x,y)
x = lda.transform(x)
print(x.shape)  # (581012, 6)

for i in range(1, x.shape[1]+1):
    lda = LinearDiscriminantAnalysis(n_components=i).fit(x,y)
    lda_x = lda.transform(x)
    
    r = int(np.random.uniform(1,1000))
    x_train, x_test, y_train, y_test = train_test_split(lda_x,y,train_size=0.7,random_state=r,stratify=y)

    #model
    model = RandomForestClassifier()

    #compile & fit
    model.fit(x_train,y_train)

    #evaluate & predict
    loss = model.score(x_test,y_test)

    print(loss)

# r=994
# LOSS: 0.1615818589925766
# ACC:  0.9583371580686616(0.9583371877670288 by loss[1])

# 0.5856549476776207
# 0.693965715072517
# 0.7608373875527813
# 0.8106354415274463
# 0.8443466587112172
# 0.8816320910592987