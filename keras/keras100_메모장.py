import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from keras.utils import to_categorical #
from imblearn.over_sampling import SMOTE
import time
start_time = time.time()

RANDOMSATAE = 12
TRAINSIZE = 0.89
VAL_SPLIT = 0.2
PATIENCE = 1000
EPOCHS = 10000
BATCHSIZE = 1024

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler() 
scaler = StandardScaler() 
# scaler = MaxAbsScaler() 
# scaler = RobustScaler() 



#1. 데이터
path = "C:/_data/dacon/dechul//"
train_csv = pd.read_csv(path + "train.csv", index_col=0 )
test_csv = pd.read_csv(path + "test.csv", index_col=0 )
submission_csv = pd.read_csv(path + "sample_submission.csv")


# 라벨 인코더
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder() 
train_csv['주택소유상태'] = le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = le.fit_transform(train_csv['대출목적'])
train_csv['대출기간'] = train_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
train_csv['근로기간'] = le.fit_transform(train_csv['근로기간'])

test_csv['주택소유상태'] = le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = le.fit_transform(test_csv['대출목적'])
test_csv['대출기간'] = test_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
test_csv['근로기간'] = le.fit_transform(test_csv['근로기간'])

train_csv['대출등급'] = le.fit_transform(train_csv['대출등급']) # 마지막에 와야함

# x와 y를 분리
x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']

y = np.reshape(y, (-1,1)) 

ohe = OneHotEncoder(sparse = False)
ohe.fit(y)
y_ohe = ohe.transform(y)
x_train, x_test, y_train, y_test = train_test_split(
            x,
            y_ohe,             
            train_size=TRAINSIZE,
            random_state=RANDOMSATAE,
            stratify=y_ohe,
            shuffle=True,
            )



scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train.shape, y_train.shape) # (82812, 13) (82812, 7)



#2. 모델 구성 

model = Sequential()
model.add(Dense(256, input_dim=13, activation='swish'))
# model.add(Dropout(0.2))
model.add(Dense(16, activation='swish'))
model.add(Dense(256, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(256, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(256, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(256, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(256, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(256, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(7, activation='softmax'))


#3.컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=PATIENCE,
                verbose=1,
                restore_best_weights=True
                )

model.fit(x_train, y_train, epochs=EPOCHS, batch_size = BATCHSIZE,
                validation_split=VAL_SPLIT,
                callbacks=[es],
                verbose=1
                )
end_time = time.time()   #끝나는 시간

#4.평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
arg_pre = np.argmax(y_predict, axis=1) 
arg_test = np.argmax(y_test, axis=1)
y_submit = model.predict(test_csv)
submit = np.argmax(y_submit, axis=1)
submitssion = le.inverse_transform(submit)
      
submission_csv['대출등급'] = submitssion
y_predict = ohe.inverse_transform(y_predict)
y_test = ohe.inverse_transform(y_test)
f1 = f1_score(y_test, y_predict, average='macro')
acc = accuracy_score(y_test, y_predict)
print("로스 : ", results[0])  
print("acc : ", results[1])  
print("f1 : ", f1)  
submission_csv.to_csv(path + f"A_dechul_240202_{acc:.4f}.csv", index=False)
print("걸린시간 : ", round(end_time - start_time, 2),"초")



