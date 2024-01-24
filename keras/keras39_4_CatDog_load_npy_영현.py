#테스트폴더 쓰진말고 train폴더로
#변환시간도 체크하기

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import os

# start_time = time.time()
path = "C:\\_data\\KAGGLE\\cat-and-dog-classification-harper2022\\"
train_path = path+"train\\"
test_path = path+"test\\"

BATCH_SIZE = int(1000)
IMAGE_SIZE = int(130)

load_path = path+f"data_{IMAGE_SIZE}px_"
x = np.load(load_path+"x.npy")
y = np.load(load_path+"y.npy")
test = np.load(load_path+"test.npy")

print(x.shape,y.shape)

r = int(np.random.uniform(1,1000))
r = 965
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=r, stratify=y)
hist = []


model = load_model("C:\_data\KAGGLE\cat-and-dog-classification-harper2022\model_save\\acc_0.852278.h5")

# 평가, 예측
# evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)
y_prediect = model.predict(test)
y_prediect = np.around(y_prediect.reshape(-1))
print(y_prediect.shape)

print(f"LOSS: {loss[0]:.6f}\nACC:  {loss[1]:.6f}")
model.save(path+f"model_save\\acc_{loss[1]:.6f}.h5")

forder_dir = path+"test\\test"
id_list = os.listdir(forder_dir)
for i, id in enumerate(id_list):
    id_list[i] = int(id.split('.')[0])

for id in id_list:
    print(id)

y_submit = pd.DataFrame({'id':id_list,'Target':y_prediect})
print(y_submit)
y_submit.to_csv(path+f"submit\\acc_{loss[1]:.6f}.csv",index=False)


import matplotlib.pyplot as plt
if hist != []:
    plt.title("Cat&Dog CNN")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.plot(hist.history['val_acc'],label='val_acc',color='red')
    plt.plot(hist.history['acc'],label='acc',color='blue')
    # plt.plot(hist.history['val_loss'],label='val_loss',color='red')
    # plt.plot(hist.history['loss'],label='loss',color='blue')
    plt.legend()
    plt.show()

# fitting time: 391.9127sec
# LOSS: 0.410849
# ACC:  0.852278