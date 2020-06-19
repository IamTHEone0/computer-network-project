import os
import cv2
import numpy as np
from decimal import *

np.set_printoptions(threshold=np.nan)    # 这里多加一行代码，避免控制台输出省略号的问题

# def check(a,b,c,d,x):
#     n =

pic_path = "H:\\data_dst\\merged\\"   # 图片路径
img = cv2.imread('3.jpg')
count = 0
l = []
for file in os.listdir(pic_path):
    file_name = pic_path + file
    img = cv2.imread(file_name)
    n = img.shape[0]*img.shape[1]
    count = 0
    for x in range(1,img.shape[0]-1):
        for y in range(1,img.shape[1]-1):
            a = img[x-1,y-1]
            b = img[x-1,y+1]
            c = img[x+1,y-1]
            d = img[x+1,y+1]
            p = a+b+c+d
            px = img[x,y]
            s = p-4*px
            if np.all(s==0):
                count += 1
    print(count)
    print(n)
    s = count/n
    print(s)
    l.append(s)
    with open("merged.txt", 'a+') as f:
        print(1)
        f.write(str(Decimal(s).quantize(Decimal('0.0000'))))
        f.write("\n")
s = 0
for i in l:
    s+=i
print(s)
print(len(l))
print(s/len(l))