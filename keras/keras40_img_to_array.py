import sys
import tensorflow as tf
print('텐서플로: ', tf.__version__) # 텐서플로:  2.9.0
print('파이썬버전: ', sys.version)  # 파이썬버전:  3.9.18



from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img # 이미지 가져옴
from tensorflow.keras.preprocessing.image import img_to_array # 이미지 수치화,/주로 한장짜리 수치화
import matplotlib.pyplot as plt
import numpy as np


path = ('c:/_data/image//cat_and_dog//train/Cat//1.jpg')
img = load_img(
    path,
    target_size = (150, 150)
)
print(img)
# <PIL.Image.Image image mode=RGB size=150x150 at 0x25E1CECFFA0>
print(type(img))    # <class 'PIL.Image.Image'>
plt.imshow(img)
plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape)    # (281, 300, 3) -> (150, 150, 3)
print(type(arr))    # <class 'numpy.ndarray'>

# 차원증가
img = np.expand_dims(arr, axis=1)   # 익스펜디드 딤 : 차원을 늘려라
print(img.shape)    # (1, 150, 150, 3)      # 액시스 1 일때 (150, 1, 150, 3)







