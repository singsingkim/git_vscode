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

#data
path = "C:\\_data\\DACON\\iris\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submit_csv = pd.read_csv(path+"sample_submission.csv")

x = train_csv.drop(['species'],axis=1)
y = train_csv['species']

print(x,y,sep='\n')
print(x.shape,y.shape)  #x:(120, 4) y:(120,) test_csv:(30, 4)

y = y.to_frame('species')                                      #pandas Series에서 dataframe으로 바꾸는 법


# print(x)
print(f"{x.shape=}, {y.shape=}")      
# print(np.unique(y, return_counts=True)) 

r = int(np.random.uniform(1,1000))
# r=326
x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r,stratify=y)

#model
model = LinearSVC(C=100)

#compile & fit
model.fit(x_train,y_train)
#evaluate & predict
loss = model.score(x_test,y_test)
print(loss)

#결과값 출력

#테스트 잘 분리되었는지 확인 및 예측결과값 비교
# print(np.unique(y_test,return_counts=True))
# print(np.unique(y_predict,return_counts=True))


# r=326
# LOSS: 0.3992086946964264
# ACC:  1.0(1.0by loss[1])

# LinearSVC
# 0.9583333333333334

# 0.5856549476776207
# 0.693965715072517
# 0.7608373875527813
# 0.8106354415274463
# 0.8443466587112172
# 0.8816320910592987