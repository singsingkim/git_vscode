import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

# 1. Data
path = "c:\\_data\\dacon\\ddarung\\"
train_csv=pd.read_csv(path + "train.csv",index_col=0)
test_csv=pd.read_csv(path + "test.csv",index_col=0)
submission_csv =pd.read_csv(path +"submission.csv")
train_csv=train_csv.dropna()
test_csv=test_csv.fillna(test_csv.mean())

x=train_csv.drop(['count'],axis=1)
y=train_csv['count']


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
    model = RandomForestRegressor(random_state=123)

    # 훈련
    model.fit(x_train, y_train)

    # 평가, 예측
    accuracy = model.score(x_test, y_test)
    print(f"Number of Components: {n}, Accuracy: {accuracy}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_components = n

print(f"\nBest Number of Components: {best_n_components}, Best Accuracy: {best_accuracy}")



# Number of Components: 9, Accuracy: 0.7016243619632474
# Number of Components: 8, Accuracy: 0.7005624355164537
# Number of Components: 7, Accuracy: 0.6976764347533392
# Number of Components: 6, Accuracy: 0.6839997042011958
# Number of Components: 5, Accuracy: 0.6839394960630181
# Number of Components: 4, Accuracy: 0.6127233367554732
# Number of Components: 3, Accuracy: 0.5897291788642296
# Number of Components: 2, Accuracy: 0.4916673150746266

# Best Number of Components: 9, Best Accuracy: 0.7016243619632474
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

