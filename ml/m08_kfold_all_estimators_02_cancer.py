import pandas as pd     # 각종 연산 ( 판다스 안의 파일들은 넘파이 형식)
from keras.models import Sequential,Model, load_model
from keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score    # rmse 사용자정의 하기 위해 불러오는것
import time        
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

print(np.unique(y, return_counts=True))
# (array([0, 1]), array([212, 357], dtype=int64))

# 넘파이 갯수 함수
print(np.count_nonzero(y==0))
print(np.count_nonzero(y==1))
print(np.count_nonzero(y==2))
# 판다스 갯수 함수
print(pd.DataFrame(y).value_counts())
print(pd.Series(y).value_counts())
print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(
                 x, y, shuffle=True, train_size= 0.7, 
                 random_state= 88888
                 )


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))  # 
print(np.min(x_test))   # 
print(np.max(x_train))  # 
print(np.max(x_test))   # 

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)



## 2. 모델구성
allAlgorithms = all_estimators(type_filter='classifier')    # 분류
# allAlgorithms = all_estimators(type_filter='regressor')   # 회귀

print('allAlgorithms', allAlgorithms)
print('모델의 갯수 :', len(allAlgorithms)) # 41 개 # 소괄호로 묶여 있으니 튜플
# 포문을 쓸수있는건 이터너리 데이터 (리스트, 튜플, 딕셔너리)
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

for name, algorithm in allAlgorithms:
    try:        
        # 2 모델
        model = algorithm()
        # 3 훈련
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print('============', name, '==============')
        print('ACC : ', scores, '\n평균 ACC : ', round(np.mean(scores), 4))
        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
        acc = accuracy_score(y_test, y_predict)
        print('cross_val_predict ACC', acc)

    except:
        print(name, ': 에러 입니다.')
        # continue # 문제가 생겼을때 다음 단계로

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold



