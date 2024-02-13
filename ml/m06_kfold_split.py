import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd

# 1 데이터
datasets = load_iris()
df = pd.DataFrame(datasets.data, columns = datasets.feature_names)

print(df)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

for train_index, val_index in kfold.split(df):
    print('='*70)
    print(train_index, '\n', val_index)
    print(len(train_index), len(val_index))
    print('훈련데이터 갯수 : ', len(train_index), '',
          '검증데이터 갯수 : ', len(val_index))

# # 2 모델
# model = SVC()


# # 3 훈련
# scores = cross_val_score(model, x, y, cv=kfold)
# print('ACC : ', scores, '\n평균 ACC : ', round(np.mean(scores), 4))


# 4 