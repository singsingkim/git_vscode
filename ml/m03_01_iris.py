import numpy as np
import pandas as pd
from sklearn.datasets import load_iris  # 꽃
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score    # rmse 사용자정의 하기 위해 불러오는것
import time            
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression # 로지스틱리그리션 = 분류다
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



# 1. 데이터
x , y = load_iris(return_X_y=True)
print(x.shape, y.shape) # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
            shuffle=True, train_size= 0.7,
            random_state= 7777,
            stratify=y,)    # 스트레티파이 와이(예스)는 분류에서만 쓴다, 트레인 사이즈에 따라 줄여주는것


## 2. 모델구성
# model = LinearSVC(C = 100)
# model = Perceptron()
# model = LogisticRegression()
model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()


# 3 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
results = model.score(x_test, y_test)
print("model.score : ", results)

y_predict = model.predict(x_test)
print(y_predict)
acc = accuracy_score(y_predict, y_test)
print('acc : ', acc)

# acc :  0.9777777777777777