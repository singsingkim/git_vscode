import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,BatchNormalization, AveragePooling1D, Flatten, Conv2D, LSTM, Bidirectional,Conv1D,MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier  # xgboost 모듈에서 XGBClassifier를 가져옵니다.

path= "c:/_data/kaggle/비만/"
train=pd.read_csv(path+"train.csv",index_col=0)
test=pd.read_csv(path+"test.csv",index_col=0)
sample=pd.read_csv(path+"sample_submission.csv")
x= train.drop(['NObeyesdad'],axis=1)
y= train['NObeyesdad']

TRAINSIZE = 0.8
RS = 7
NUM = 41

SAVENAME = f'biman_xgb_{NUM}'

lb = LabelEncoder()

y = lb.fit_transform(train['NObeyesdad'])

# lb.classes_ 에는 라벨 순서가 들어있습니다.
# 예를 들어 lb.classes_ 가 ['Insufficient_Weight', 'Normal_Weight', ...] 인 경우,
# Insufficient_Weight는 0, Normal_Weight는 1, ... 로 매핑됩니다.


# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']

# 데이터프레임 x의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])

# 데이터프레임 test_csv의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(test[column])
    test[column] = lb.transform(test[column])

x_train,x_test,y_train,y_test=train_test_split(
    x,y,train_size=TRAINSIZE,random_state=RS,stratify=y)

import random

r = random.randint(1, 1000)
random_state = r

# XGBoost 분류기를 위한 매개변수를 정의합니다.
xgb_params = {
    'objective': 'multi:softmax',  # 다중 분류
    'num_class': 7,  # 클래스의 개수
    'learning_rate': 0.1,  # 학습률
    'max_depth': 6,  # 트리의 최대 깊이
    'min_child_weight': 1,  # 자식 노드의 최소 가중치 합
    'subsample': 0.8,  # 데이터 샘플링 비율
    'colsample_bytree': 0.8,  # 트리를 생성할 때 특성 샘플링 비율
    'gamma': 0,  # 리프 노드를 추가적으로 나눌 최소 손실 감소량
    'reg_alpha': 0,  # L1 정규화 가중치
    'reg_lambda': 1,  # L2 정규화 가중치
    'random_state': random_state  # 랜덤 시드 값
}

# 정의된 매개변수로 XGBClassifier를 인스턴스화합니다.
model = XGBClassifier(**xgb_params)




# 모델을 훈련 데이터에 맞춥니다.
model.fit(x_train, y_train)

# 훈련된 모델을 저장합니다.
model.save_model(f"c:\_data\_save\\{SAVENAME}.h5")

# 테스트 데이터에 대한 예측을 수행합니다.
y_pred = model.predict(x_test)
y_submit = model.predict(test)
sample['NObeyesdad'] = y_submit

# 결과를 CSV 파일로 저장합니다.
sample.to_csv(path + f"{SAVENAME}.csv", index=False)

# 정확도를 평가합니다.
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("r", r)
