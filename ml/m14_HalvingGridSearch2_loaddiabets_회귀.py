import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV
# 익스페리먼트 이네이블과 쌍으로 사용해야한다
import time

# 1 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8,
    )

print(x.shape)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {'C':[1, 10, 100, 1000], 'kernel':['linear'], 'degree':[3, 4, 5]},  # 6 번
    {'C':[1, 10, 100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},      # 12번
    {'C':[1, 10, 100, 1000], 'kernel':['sigmoid'],                      # 24번
            'gamma':[0.01, 0.001, 0.0001], 'degree':[3, 4]},
]

# 그리드로 돌렸을때 5 * 42 횟수.

# 2 모델
# model = SVC(C=1, kernel='linear', degree=3)
# model = GridSearchCV(SVC(), 
#                      parameters,
#                      cv = kfold,
#                      verbose=1,
#                     #  refit=True # 디폴트 트루 # 한바퀴 돌린후 다시 돌린다
#                      n_jobs=3   # 24개의 코어중 3개 사용 / 전부사용 -1
#                      )

# model = RandomizedSearchCV(SVC(), 
print('==============하빙그리드서치 시작==========================')
model = HalvingGridSearchCV(SVR(), 
                     parameters,
                     cv = kfold,
                     verbose=1,
                    #  refit=True # 디폴트 트루 # 한바퀴 돌린후 다시 돌린다
                     n_jobs=3,   # 24개의 코어중 3개 사용 / 전부사용 -1
                     random_state=123,
                     factor=3
                    #  min_resources=150
                     )




start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print('최적의 매개변수 : ', model.best_estimator_)
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
print('최적의 파라미터 : ', model.best_params_) # 내가 선택한것
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'} 우리가 지정한거중에 가장 좋은거
print('best_score : ', model.best_score_)   # 핏한거의 최고의 스코어
# best_score :  0.975
print('model_score : ', model.score(x_test, y_test))    # 
# model_score :  0.9666666666666667


y_predict = model.predict(x_test)
print('r2_score', r2_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
            # SVC(C-1, kernel='linear').predicict(x_test)
print('최적 튠 R2 : ', r2_score(y_test, y_pred_best))

print('걸린신간 : ', round(end_time - start_time, 2), '초')

# import pandas as pd
# print(pd.DataFrame(model.cv_results_).T)


# ==============하빙그리드서치 시작==========================
# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 13
# max_resources_: 353
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 42
# n_resources: 13           회귀// cv * 2 + a(알파)
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# c:\Users\bitcamp\anaconda3\envs\tf209gpu\lib\site-packages\sklearn\model_selection\_split.py:684: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
#   warnings.warn(
# ----------
# iter: 1
# n_candidates: 14
# n_resources: 39
# Fitting 5 folds for each of 14 candidates, totalling 70 fits
# c:\Users\bitcamp\anaconda3\envs\tf209gpu\lib\site-packages\sklearn\model_selection\_split.py:684: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
#   warnings.warn(
# ----------
# iter: 2
# n_candidates: 5
# n_resources: 117
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# c:\Users\bitcamp\anaconda3\envs\tf209gpu\lib\site-packages\sklearn\model_selection\_split.py:684: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
#   warnings.warn(
# ----------
# iter: 3
# n_candidates: 2
# n_resources: 351
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# c:\Users\bitcamp\anaconda3\envs\tf209gpu\lib\site-packages\sklearn\model_selection\_split.py:684: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
#   warnings.warn(
# 최적의 매개변수 :  SVR(C=10, degree=4, kernel='linear')
# 최적의 파라미터 :  {'C': 10, 'degree': 4, 'kernel': 'linear'}
# best_score :  0.11105690406318194
# model_score :  0.1449286359325247
# r2_score 0.1449286359325247
# 최적 튠 R2 :  0.1449286359325247
# 걸린신간 :  0.67 초
# PS C:\Study>