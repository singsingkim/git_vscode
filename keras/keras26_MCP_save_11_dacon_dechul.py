from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "c://_data//dacon//dechul//"

train_csv=pd.read_csv(path+"train.csv",index_col=0)

test_csv=pd.read_csv(path+"test.csv",index_col=0)

sub_csv=pd.read_csv(path+"sample_submission.csv")

# train_csv=train_csv[train_csv['근로기간'] != 'Unknown']

# unique,count = np.unique(train_csv['근로기간'], return_counts=True)

le=LabelEncoder()
train_le=LabelEncoder()
test_le=LabelEncoder()





train_csv['근로기간']=train_le.fit_transform(train_csv['근로기간'])
train_csv['주택소유상태']=train_le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적']=train_le.fit_transform(train_csv['대출목적'])
train_csv['대출등급']=train_le.fit_transform(train_csv['대출등급'])

test_csv['근로기간']=test_le.fit_transform(test_csv['근로기간'])
test_csv['주택소유상태']=test_le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적']=test_le.fit_transform(test_csv['대출목적'])


train_csv['대출기간'] = train_csv['대출기간'].str.split().str[0].astype(int)
test_csv['대출기간'] = test_csv['대출기간'].str.split().str[0].astype(int)
# train_csv['대출기간']=train_le.fit_transform(train_csv['대출기간'])
# test_csv['대출기간']=test_le.fit_transform(test_csv['대출기간'])





x=train_csv.drop('대출등급',axis=1)
y=train_csv['대출등급']

train_csv.dropna

# print(np.unique(y)) #['A' 'B' 'C' 'D' 'E' 'F' 'G']
# print(pd.value_counts(y))
# B    28817
# C    27623
# A    16772
# D    13354
# E     7354
# F     1954
# G      420

# print(train_csv.head(8))
yo = to_categorical(y)

print(x.shape,yo.shape) #(96294, 13) (96294, 7)

x_train,x_test,y_train,y_test=train_test_split(x,yo,train_size=0.8,
                                               random_state=112,stratify=y)



from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler


# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 
# print(np.min(x_test))   # 
# print(np.max(x_train))  # 
# print(np.max(x_test))   # 

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 
# print(np.min(x_test))   # 
# print(np.max(x_train))  # 
# print(np.max(x_test))   # 

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 
# print(np.min(x_test))   # 
# print(np.max(x_train))  # 
# print(np.max(x_test))   # 

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))  # 
print(np.min(x_test))   # 
print(np.max(x_train))  # 
print(np.max(x_test))   # 


model=Sequential()
model.add(Dense(32,input_shape=(13,)))
model.add(Dense(64,))
model.add(Dense(128,))
model.add(Dense(64,))
model.add(Dense(32,))
model.add(Dense(16,))
model.add(Dense(8,))
model.add(Dense(7,activation='softmax'))
# 3
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
print(date)         # 2024-01-17 10:54:10.769322
print(type(date))   # <class 'datetime.datetime')
date = date.strftime("%m%d_%H%M")
print(date)         # 0117_1058
print(type(date))   # <class 'str'>

path='..\_data\_save\MCP\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0~9999 에포 , 0.9999 발로스
filepath = "".join([path,'k26_11_dacon_dechul_', date,'_', filename])
# '..\_data\_save\MCP\\k25_0117_1058_0101-0.3333.hdf5'


es=EarlyStopping(monitor='val_acc',mode='min',patience=200,verbose=1,
                 restore_best_weights=True)

mcp = ModelCheckpoint(
    monitor='val_loss', mode = 'auto', verbose=1,save_best_only=True,
    filepath=filepath
    )

hist=model.fit(x_train,y_train,epochs=4000,batch_size=500,validation_split=0.2,callbacks=[es,mcp])



# 4
result=model.evaluate(x_test,y_test)
print("loss",result[0])
print("acc",result[1])
y_predict=model.predict(x_test)

arg_y_test=np.argmax(y_test,axis=1)
arg_y_predict=np.argmax(y_predict,axis=1)
f1_score=f1_score(arg_y_test,arg_y_predict,average='macro')
print("f1_score:",f1_score)
y_submit=np.argmax(model.predict(test_csv),axis=1)
y_submit=train_le.inverse_transform(y_submit)


sub_csv['대출등급']=y_submit
sub_csv.to_csv(path+"sample_submission_0116_4.csv",index=False)


# 민맥스스케일
# loss 1.5657554864883423
# acc 0.331481397151947
# 602/602 [==============================] - 0s 443us/step
# f1_score: 0.11656874545039399

# 맥스앱스스케일
# loss 1.5781593322753906
# acc 0.3204735517501831
# 602/602 [==============================] - 0s 474us/step
# f1_score: 0.1151665256379172

# 스탠다드스케일
# loss 1.540550708770752
# acc 0.33241602778434753
# 602/602 [==============================] - 0s 471us/step
# f1_score: 0.14066193829440912

# 로부투스스케일
# loss 1.6365586519241333
# acc 0.29523858428001404
# 602/602 [==============================] - 0s 487us/step
# f1_score: 0.07365918261212859