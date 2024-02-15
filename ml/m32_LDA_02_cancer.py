import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# data
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets.data
y = datasets.target 
df_y = pd.DataFrame(y)

x = StandardScaler().fit_transform(x,y)
lda = LinearDiscriminantAnalysis().fit(x,y)
x = lda.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,shuffle=False)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#model 
model = RandomForestClassifier()
#compile & fit
model.fit(x_train,y_train)

#evaluate & predict
loss = model.score(x_test,y_test)
y_predict = model.predict(x_test)
# accuracy
# print(y_test, np.around(y_predict,0))
def ACC(y_true, y_predict):
    return accuracy_score(y_true, np.around(y_predict))
acc =ACC(y_test,y_predict)
print(acc)
print(f"ACCURACY: {loss}")

# MinMaxScaler
# ACCURACY: 0.9707602262496948

# ACCURACY: 0.9590643274853801