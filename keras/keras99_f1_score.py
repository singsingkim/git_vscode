import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from keras.models import Sequential, Model
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
# 1 데이터
datasets = load_wine()

x = datasets.data
y = datasets['target']  # x 와 같다
print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

# x y 값 줄인다
# 일부러 데이터 불균형하게 만든것 
# 증폭을 하기위해 데이터를 일부러 죽임
x = x[:-35]
y = y[:-35]
print(y)
print(np.unique(y, return_counts=True))
# 분류형 라벨 3개
# (array([0, 1, 2]), array([59, 71, 13], dtype=int64))  

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=123,
    stratify=y  #y의 라벨 갯수만큼 비율로 자른다
)

# 2 모델
model = Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))

# 3 컴파일, 훈련
# 원핫을 하기싫으면 loss 에 sparse 를 넣어준다
# 원핫 효과를 대체한다
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=200, validation_split=0.2)

# 4 평가, 예측
results = model.evaluate(x_test, y_test)

# 지표 : f1_score
y_predict = model.predict(x_test)
print(y_test)       # 원핫 안돼있음
print(y_predict)    # 원핫 돼잇음
# 원핫 되어 있는걸 원핫을 풀어야한다
# 가장높은숫자의 위치를 뺀다
y_predict = np.argmax(y_predict, axis=1)
# 행에서 가장 큰놈 뺀다
print(y_predict)

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)

# f1 은 원래 이진분류에서 사용한다. 
f1 = f1_score(y_test, y_predict, average='macro')
print('f1 : ', f1)

# 데이터 감축 한 상태에서의 결과
# acc :  0.8333333333333334
# f1 :  0.7352941176470589






