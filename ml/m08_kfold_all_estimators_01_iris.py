import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


# 1 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, stratify=y
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


# 2 모델

## 2. 모델구성
allAlgorithms = all_estimators(type_filter='classifier')    # 분류
# allAlgorithms = all_estimators(tydpe_filter='regressor')   # 회귀

print('allAlgorithms', allAlgorithms)
print('모델의 갯수 :', len(allAlgorithms)) # 41 개 # 소괄호로 묶여 있으니 튜플
# 포문을 쓸수있는건 이터너리 데이터 (리스트, 튜플, 딕셔너리)
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

for name, algorithm in allAlgorithms:
    try:        
        # 2 모델
        model = algorithm()
        # 3 훈련
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print('============', name, '==============')
        print('ACC : ', scores, '\n평균 ACC : ', round(np.mean(scores), 4))
        
        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
        acc = accuracy_score(y_test, y_predict)
        print('cross_val_predict ACC', acc)
    except:
        print(name, ': 에러 입니다.')
        # continue # 문제가 생겼을때 다음 단계로





