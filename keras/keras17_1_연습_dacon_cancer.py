# 17_1 카피
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import time

# 1. 데이터
path = "C:\_data\dacon\cancer _연습\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

print(train_csv)        # [652 rows x 9 columns] 인덱스 삭제 안하면 10 컬럼이다
print(test_csv)         # [116 rows x 8 columns]
print(submission_csv)   # [116 rows x 2 columns]

print(train_csv.info()) # 컬럼을 행으로 타입과 형태를 보여줌
print(train_csv.describe())

# train_csv = train_csv.dropna()        # 결측치 포함 행 삭제  
# train_csv = train_csv.fillna(0)       # 결측치 데이터 0 으로
print(train_csv.isna().sum())           # 결측 데이터의 합

# test_csv = test_csv.fillna(test_csv.mean()) # 널값에 평균을 넣음
# TEST_115            2       84              0              0        0   0.0                     0.304   21

# print(test_csv)
# TEST_115            2       84              0              0        0   0.0                     0.304   21

########## x 와 y 를 분리 ##########
x = train_csv.drop(['Outcome'], axis = 1)
y = train_csv['Outcome']

print(np.unique(y, return_counts=True))
# (array([0, 1], dtype=int64), array([424, 228], dtype=int64))
print(np.count_nonzero(y==0))   # 424
print(np.count_nonzero(y==1))   # 228
print(np.count_nonzero(y==2))   # 0

print(pd.DataFrame(y).value_counts())
# Outcome
# 0          424
# 1          228
# Name: count, dtype: int64
print(pd.Series(y).value_counts())
# Outcome
# 0    424
# 1    228
# Name: count, dtype: int64
print(pd.value_counts(y))
# Outcome
# 0    424
# 1    228
# Name: count, dtype: int64

x_train, x_test, y_train, y_test = train_test_split(
            x, y, shuffle=True, train_size=0.7,
            random_state=585858
            )

print(x_train.shape, x_test.shape)  # (456, 8) (196, 8)
print(y_train.shape, y_test.shape)  # (456,) (196,)

# 2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim = 8))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1, activation = 'sigmoid'))     
# 시그모이드를 사용하면 0 ~ 1 사이의 값이 나온다.안그러면 0 ~ 1 바깥으로 값이 튄다
# 다중분류 일때는 카테고리 크로스엔트로피를 사용할때는 소프트맥스를 사용한다
# 이진분류 일때는 바이너리 크로스엔트로피를 사용할때는 시그모이드를 사용한다.


# 3. 컴파일, 훈련
model



