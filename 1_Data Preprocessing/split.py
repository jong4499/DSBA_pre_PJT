import cv2
import os
import shutil
import torch
from IPython.display import Image, clear_output
from glob import glob
from sklearn.model_selection import train_test_split

img_lst = glob('C:/Users/admin/Desktop/data_V4/images/*.jpg')
print("총 이미지 수: ", len(img_lst))
label_lst = glob('C:/Users/admin/Desktop/data_V4/labels/*.xml')
print("총 라벨 수: ", len(label_lst))

# split data------------------------------------------------------------------------------------


x_train, x_val, y_train, y_val = train_test_split(img_lst, label_lst, test_size=0.2, random_state=555)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=555)

print("-"*50)
print("\n데이터셋 분할 완료")

print("x train length: ", len(x_train))
print("x val length: ", len(x_val))
print("x test length: ", len(x_test))

print("y train length: ", len(y_train))
print("y val length: ",len(y_val))
print("y test length: ",len(x_test))

print("\n")
print("-"*50)
print("\n")


# move data------------------------------------------------------------------------------------


os.mkdir('C:/Users/admin/Desktop/data_V4/images/train')
os.mkdir('C:/Users/admin/Desktop/data_V4/images/val')
os.mkdir('C:/Users/admin/Desktop/data_V4/images/test')

os.mkdir('C:/Users/admin/Desktop/data_V4/labels/train')
os.mkdir('C:/Users/admin/Desktop/data_V4/labels/val')
os.mkdir('C:/Users/admin/Desktop/data_V4/labels/test')


for path in x_train:
  shutil.move(path, 'C:/Users/admin/Desktop/data_V4/images/train')
for path in x_val:
  shutil.move(path, 'C:/Users/admin/Desktop/data_V4/images/val')
for path in x_test:
  shutil.move(path, 'C:/Users/admin/Desktop/data_V4/images/test')

for path in y_train:
  shutil.move(path, 'C:/Users/admin/Desktop/data_V4/labels/train')
for path in y_val:
  shutil.move(path, 'C:/Users/admin/Desktop/data_V4/labels/val')
for path in y_test:
  shutil.move(path, 'C:/Users/admin/Desktop/data_V4/labels/test')

print("-"*50)
print("\n데이터셋 이동 완료")

print("train img data's length: ", len(glob('C:/Users/admin/Desktop/data_V4/images/train/*.jpg')))
print("validation img data's length: ", len(glob('C:/Users/admin/Desktop/data_V4/images/val/*.jpg')))
print("test img data's length: ", len(glob('C:/Users/admin/Desktop/data_V4/images/test/*.jpg')))

print("train label data's length: ", len(glob('C:/Users/admin/Desktop/data_V4/labels/train/*.txt')))
print("validation label data's length: ",len(glob('C:/Users/admin/Desktop/data_V4/labels/val/*.txt')))
print("test label data's length: ", len(glob('C:/Users/admin/Desktop/data_V4/labels/test/*.txt')))

print("\n")
print("-"*50)
print("\n")


# (train, val, test).txt------------------------------------------------------------------------------------


train_img_lst = glob('C:/Users/admin/Desktop/data_V4/images/train/*.jpg')
val_img_lst = glob('C:/Users/admin/Desktop/data_V4/images/val/*.jpg')
test_img_lst = glob('C:/Users/admin/Desktop/data_V4/images/test/*.jpg')

with open('C:/Users/admin/Desktop/data_V4/train.txt', 'w')as f:
  f.write('\n'.join(train_img_lst) + '\n')

with open('C:/Users/admin/Desktop/data_V4/val.txt', 'w')as f:
 f.write('\n'.join(val_img_lst ) + '\n')

with open('C:/Users/admin/Desktop/data_V4/test.txt', 'w')as f:
 f.write('\n'.join(test_img_lst) + '\n')

print("-"*50)
print("txt 파일 생성 완료")
print("데이터셋 처리가 마무리 되었습니다")