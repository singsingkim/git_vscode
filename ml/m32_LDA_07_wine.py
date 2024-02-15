from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import sklearn.preprocessing
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

path = "C:\\_data\\DACON\\와인품질분류\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submit_csv = pd.read_csv(path+"sample_submission.csv")

# print(train_csv.isna().sum(),test_csv.isna().sum()) 결측치 존재안함

# print(train_csv,test_csv,sep='\n') #[5497 rows x 13 columns], [1000 rows x 12 columns]
x = train_csv.drop(['quality'],axis=1)
y = train_csv['quality']

print(np.unique(y,return_counts=True))

# print(x.shape,y.shape)  #(5497, 12) (5497,)
print(np.unique(y,return_counts=True))
# print(y.shape)          #(5497, 7)

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
# print(x)
test_csv.loc[test_csv['type'] == 'red', 'type'] = 1 
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

x = StandardScaler().fit_transform(x)
lda = LinearDiscriminantAnalysis().fit(x,y)
x = lda.transform(x)
print(x.shape)  # (5497, 6)

acc_list = []
for i in range(1,x.shape[1]+1):
    r = 894
    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r,stratify=y)

    #model
    model = RandomForestClassifier()

    #compile & fit
    model.fit(x_train,y_train)

    #evaluate & predict
    loss = model.score(x_test,y_test)
    print(loss)
    acc_list.append(loss)
    
print(acc_list)


# r=894
# LOSS: 2.3342490196228027
#  ACC:  0.5781818181818181(0.578181803226471 by loss[1])

# r=912
# LOSS: 1.0722206830978394
#  ACC:  0.5536363636363636(0.553636372089386 by loss[1])

# r=20
# LOSS: 1.0789892673492432
#  ACC:  0.5509090909090909(0.5509091019630432 by loss[1])

# r=433
# LOSS: 1.0544521808624268
#  ACC:  0.5572727272727273(0.557272732257843 by loss[1])

# MinMaxScaler
# LOSS: 1.0109286308288574
#  ACC:  0.5690909090909091(0.5690909028053284 by loss[1])

# StandardScaler
# LOSS: 1.0394665002822876
#  ACC:  0.5745454545454546(0.5745454430580139 by loss[1])

# MaxAbsScaler
# LOSS: 1.0278842449188232
#  ACC:  0.56(0.5600000023841858 by loss[1])

# RobustScaler
# LOSS: 1.058943510055542
#  ACC:  0.5636363636363636(0.5636363625526428 by loss[1])

# LinearSVC
# 0.42363636363636364

# [0.6718181818181819, 0.6809090909090909, 0.6727272727272727, 0.66, 0.68, 0.6772727272727272]