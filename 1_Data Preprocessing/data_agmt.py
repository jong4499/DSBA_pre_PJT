import os
import cv2
import glob
import shutil
import random
import numpy as np
import PIL.ImageOps   
from PIL import Image
import time


def agmt(src1, src2): 
    src_path = src1
    save_path = src2

    file_names = os.listdir(src_path)
    total_origin_image_num = len(file_names) - 1

    agmt_cnt = 1

    for i in range(1, total_origin_image_num):
        word_ttp = file_names[i].split('_')[0]
        pic_num = file_names[i].split('_')[1]
        img = cv2.imread(src_path + file_names[i])

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bright = cv2.subtract(gray, 50)
        blur = cv2.GaussianBlur(bright,(15,15),0)

        cv2.imwrite(f"{save_path}gray_{word_ttp}_{pic_num.split('.')[0]}.jpg", blur)
        agmt_cnt += 1
        print("converted.  cnt:", (agmt_cnt))

    print('증강 작업 완료')



def rename(path):
    file_path = os.listdir(path)
    total_origin_image_num = len(file_path) - 1

    for i in range(1, total_origin_image_num):
        word_ttp = file_path[i].split('_')[0]
        pic_num = file_path[i].split('_')[1]
        os.rename(path+file_path[i], f"{path}gray_{word_ttp}_{pic_num.split('.')[0]}.txt")

    print('이름변경 작업 완료')
    


