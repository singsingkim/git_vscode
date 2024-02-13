# 14_3 카피
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Flatten, Conv1D
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score    # r2 결정계수
import numpy as np
import time
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y,
            train_size = 0.7,
            test_size = 0.3,
            shuffle = True,
            random_state = 4567)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler


scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))  # 
print(np.min(x_test))   # 
print(np.max(x_train))  # 
print(np.max(x_test))   # 


print(x)
print(y)
print(x.shape, y.shape)     # (442, 10) (442,)
print(datasets.feature_names)   # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)

x = x.reshape(-1, 10, 1)


from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)


## 2. 모델구성
allAlgorithms = all_estimators(type_filter='classifier')    # 분류
# allAlgorithms = all_estimators(type_filter='regressor')   # 회귀

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







# #3. 컴파일, 훈련
# scores = cross_val_score(model, x_train, y_train, cv=kfold)
# print('ACC : ', scores, '\n평균 ACC : ', round(np.mean(scores), 4))

# model.fit(x_train, y_train)


# #4. 평가, 예측

# results = model.score(x_test, y_test)
# y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)    # 결정계수


# print("acc : ", results)
# print("R2 스코어 : ", r2)


