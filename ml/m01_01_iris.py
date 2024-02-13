import numpy as np
import pandas as pd
from sklearn.datasets import load_iris  # 꽃
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score    # rmse 사용자정의 하기 위해 불러오는것
import time            
from sklearn.svm import LinearSVC



# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, 
            shuffle=True, train_size= 0.7,
            random_state= 7777,
            stratify=y,)    # 스트레티파이 와이(예스)는 분류에서만 쓴다, 트레인 사이즈에 따라 줄여주는것

print(np.unique(y_test, return_counts=True))

## 2. 모델구성
# 회귀모델 : LinearSVR() - r2
# 분류모델 : LinearSVC() - acc, f1
# 기본 추출하는것 : acc
# LinearSVC() 파라미터 C 가 크면 training 포인트를 정확히 구분(굴곡지다)
#                      C 가 작으면 직선에 가깝다

model = LinearSVC(C = 100)


# 3 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
results = model.score(x_test, y_test)
print("model.score : ", results)

y_predict = model.predict(x_test)
print(y_predict)
acc = accuracy_score(y_predict, y_test)
print('acc : ', acc)
