import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
import time
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')




# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (178, 13) (178,)
print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
print(y)

print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# 0 은 59개
# 1 은 71개
# 2 는 48개
print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
# ============================================
# # ========== 원 핫 인코딩 전처리 ==============
# # 1) 케라스
# from keras.utils import to_categorical
# y_ohe = to_categorical(y)   # [1. 0. 0. ] 으로 표현
# print(y_ohe)
# print(y_ohe.shape)  # (178, 3)

# # 2) 판다스
# y_ohe2 = pd.get_dummies(y)  # [True  False  False] 으로 표현
# print(y_ohe2)
# print(y_ohe2.shape) # (178, 3)

# # 3) 사이킷런
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()   # (sparse=False)
# y = y.reshape(-1, 1)    # (행, 열) 형태로 재정의 // -1 은 열의 정수값에 따라 알아서 행을 맞추어 재정의하라 
# y_ohe3 = ohe.fit_transform(y).toarray() # // 투어레이 사용하면 위에 스파라스 안씀. 스파라스 사용하면 투어레이 안씀
# print(y_ohe3)
# print(y_ohe3.shape) # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
            shuffle=True, train_size= 0.7,
            random_state= 7777,
            stratify=y,)    # 스트레티파이 와이(예스)는 분류에서만 쓴다, 트레인 사이즈에 따라 줄여주는것


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


print(np.unique(y_test, return_counts=True))
# (array([0., 1.], dtype=float32), array([108,  54], dtype=int64))

print(x)
print(y)


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
        model.fit(x_train, y_train)
        
        acc = model.score(x_test, y_test)
        print(name, '의 정답률 :', acc)
    except:
        print(name, ': 에러 입니다.')
        # continue # 문제가 생겼을때 다음 단계로

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


model = SVC()


#3. 컴파일, 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('ACC : ', scores, '\n평균 ACC : ', round(np.mean(scores), 4))

model.fit(x_train, y_train)


#4. 평가, 예측
results = model.score(x_test, y_test)
print('results', results)
