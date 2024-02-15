from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#data
path = "C:\\_data\\DACON\\diabets\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0).drop(['Insulin','SkinThickness'],axis=1)
submission_csv = pd.read_csv(path+"sample_submission.csv")

print(train_csv.isna().sum(), test_csv.isna().sum()) # 둘다 결측치 존재하지 않음

test = train_csv['BloodPressure']#.where(train_csv['BloodPressure']==0,train_csv['BloodPressure'].mean())

    
train_csv['BloodPressure'] = test

x = train_csv.drop(['Outcome','Insulin','SkinThickness'],axis=1)
y = train_csv['Outcome']

x = StandardScaler().fit_transform(x)
lda = LinearDiscriminantAnalysis().fit(x,y)
x = lda.transform(x)
print(x.shape)  # (652, 1)
erv_sum = np.cumsum(lda.explained_variance_ratio_)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,shuffle=False)

#model
model = RandomForestClassifier()

#compile & fit
model.fit(x_train,y_train)

#evaluate & predict
loss = model.score(x_test,y_test)

print(f"ACC: {loss}\n")

# RobustScaler
# LOSS: 0.58819979429245
# ACC: 0.7448979616165161

# ACC: 0.673469387755102