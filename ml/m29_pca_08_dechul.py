import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
# 1. Data
path = "c://_data//dacon//wine//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv",index_col=0)
# print(test_csv)
submission_csv=pd.read_csv(path + "sample_submission.csv")

train_csv['type']=train_csv['type'].map({'white':1,'red':0}).astype(int)
test_csv['type']=test_csv['type'].map({'white':1,'red':0}).astype(int)

x=train_csv.drop(['quality'],axis=1)
y=train_csv['quality']

label=LabelEncoder()
label.fit(y)
y=label.transform(y)


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

best_accuracy = 0
best_n_components = 0

for n in range(x.shape[1], 1, -1):
    pca = PCA(n_components=n)
    x_pca = pca.fit_transform(x_scaled)

    x_train, x_test, y_train, y_test = train_test_split(
        x_pca, y, train_size=0.8, random_state=123, shuffle=True, # stratify=y
    )

    # 모델
    model = RandomForestClassifier(random_state=123)

    # 훈련
    model.fit(x_train, y_train)

    # 평가, 예측
    accuracy = model.score(x_test, y_test)
    print(f"Number of Components: {n}, Accuracy: {accuracy}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_components = n

print(f"\nBest Number of Components: {best_n_components}, Best Accuracy: {best_accuracy}")


# Number of Components: 12, Accuracy: 0.6590909090909091
# Number of Components: 11, Accuracy: 0.6554545454545454
# Number of Components: 10, Accuracy: 0.6518181818181819
# Number of Components: 9, Accuracy: 0.6372727272727273
# Number of Components: 8, Accuracy: 0.6454545454545455
# Number of Components: 7, Accuracy: 0.6372727272727273
# Number of Components: 6, Accuracy: 0.6327272727272727
# Number of Components: 5, Accuracy: 0.6190909090909091
# Number of Components: 4, Accuracy: 0.62
# Number of Components: 3, Accuracy: 0.5754545454545454
# Number of Components: 2, Accuracy: 0.5381818181818182

# Best Number of Components: 12, Best Accuracy: 0.6590909090909091
# PS C:\Study> 


evr = pca.explained_variance_ratio_     # 1.0 에 가까워야 좋다
print(evr)
print(sum(evr))

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()

