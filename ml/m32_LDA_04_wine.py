from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape,y.shape)  #(178, 13) (178,)
# print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48

x = StandardScaler().fit_transform(x)
lda = LinearDiscriminantAnalysis().fit(x,y)
x = lda.transform(x)
print(x.shape)  # (178, 2)

r = 398
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r,stratify=y)

#model
model = RandomForestClassifier()

#compile & fit
model.fit(x_train,y_train)

#evaluate & predict
loss = model.score(x_test,y_test)

print(f"{r=} \nACC:  {loss}")


# LinearSVC
# r=398 
# ACC:  1.0

# ACC:  1.0