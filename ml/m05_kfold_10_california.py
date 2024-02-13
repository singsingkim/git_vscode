# 14_2 카피
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv1D, Flatten, SimpleRNN
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import time
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')



#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)
print(datasets.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)   # 20640 행 , 8 열

# [실습] 만들기
# R2 0.55 ~ 0.6 이상

x_train, x_test, y_train, y_test = train_test_split(x, y,
            train_size = 0.7,
            test_size = 0.3,
            shuffle = True,
            random_state = 4567)

print(x_train.shape, y_train.shape) # (14447, 8) (14447,)
# x_train = x_train.reshape(-1,8,1)
# x_test = x_test.reshape(-1,8,1)

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
from sklearn.model_selection import StratifiedKFold
n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


model = SVR()




# # 3 컴파일, 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('ACC : ', scores, '\n평균 ACC : ', round(np.mean(scores), 4))

model.fit(x_train, y_train)

# 4 예측 평가
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("acc : ", results)
print("R2 스코어 : ", r2)

