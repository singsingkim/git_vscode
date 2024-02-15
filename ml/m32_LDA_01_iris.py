from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

#data
datasets = load_iris()

x = datasets.data
y = datasets.target

# print(x)
print(f"{x.shape=}, {y.shape=}")        #x.shape=(150, 4), y.shape=(150,)
# print(np.unique(y, return_counts=True)) #array([0, 1, 2]), array([50, 50, 50])

x = StandardScaler().fit_transform(x)
lda = LinearDiscriminantAnalysis().fit(x,y)
x = lda.transform(x)
print(x.shape)  # (150, 2)

x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=326,stratify=y)

#model
model = RandomForestClassifier()

#compile & fit
model.fit(x_train,y_train)

#evaluate & predict
loss = model.score(x_test,y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)

#결과값 출력
print(f"ACC: {acc}")
# ACC: 1.0


# SVM
# r=326
# ACC: 1.0