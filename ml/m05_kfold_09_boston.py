# 드롭 아웃 
import warnings
import numpy as np
import pandas as pd
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Dropout, Input, Conv2D, Flatten, Conv1D
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


# 1 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, 
    shuffle=True, random_state=4567)

print(x.shape, y.shape) # (506, 13) (506,)

print(x_train.shape, y_train.shape) # (1, 13) (1,)
print(x_test.shape, y_test.shape) # (505, 13) (505,)
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)
print(x_train.shape)    # (354, 13, 1, 1)
print(x_test.shape)     # (152, 13, 1, 1)

print(x_train[0])
print(y_train[0])
# unique, count = np.unique(y_train, return_counts=True)
print(y_train)
# print(unique, count)


print(np.unique(x_train, return_counts=True))
print(y_train.shape, y_test.shape)
# print(pd.value_counts(x_train))
# ohe = OneHotEncoder()
# y_train = ohe.fit_transform(y_train.reshape(-1,1)).toarray()
# y_test = ohe.fit_transform(y_test.reshape(-1,1)).toarray()
print(x_train.shape, y_train.shape) # (354, 13, 1, 1) (354,)


## 2. 모델구성
# allAlgorithms = all_estimators(type_filter='classifier')    # 분류
allAlgorithms = all_estimators(type_filter='regressor')   # 회귀

print('allAlgorithms', allAlgorithms)
print('모델의 갯수 :', len(allAlgorithms)) # 55 개 # 소괄호로 묶여 있으니 튜플
# 포문을 쓸수있는건 이터너리 데이터 (리스트, 튜플, 딕셔너리)
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

for name, algorithm in allAlgorithms:
    try:        
        # 2 모델
        model = algorithm()
        # 3 훈련
        model.fit(x_train, y_train)
        
        acc = model.score(x_test, y_test)
        print(name, '의 정답률 :', acc)
    except:
        print(name, ': 에러 입니다.')
        # continue # 문제가 생겼을때 다음 단계로

from sklearn.svm import SVR
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)


model = SVR()


scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('ACC : ', scores, '\n평균 ACC : ', round(np.mean(scores), 4))


# # # 3 훈련
model.fit(x_train, y_train)


# 4 예측, 평가
results = model.score(x_test, y_test)
print("acc : ", results)

y_predict = model.predict(x_test)
r2 = r2_score(y_predict, y_test)
print("R2 스코어 : ", r2)

