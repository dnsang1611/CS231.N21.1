import cv2 
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import numpy as np
import math
import time
import imageio

listImg = os.listdir('gif')
gifList = []
for imgPath in listImg:
    g = imageio.mimread(f'gif/{imgPath}')
    gifList.append(g)

bgr_gifList = []
for i in range(len(gifList)):
    bgr_frames = []
    for frame in gifList[i]:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bgr_frames.append(bgr_frame)
    bgr_gifList.append(bgr_frames)  


def find_object_edges(condition_matrix, iter):
    # Chuyển ma trận điều kiện thành ma trận nhị phân
    binary_matrix = np.where(condition_matrix, 1, 0).astype(np.uint8)
    # Áp dụng Canny Edge Detection
    edges = cv2.Canny(binary_matrix, 0, 1)
    dilated_edges = cv2.dilate(edges, None, iterations=iter) 
    object_edges = dilated_edges.astype(bool)
    # Tìm vị trí của các điểm biên
    return object_edges


def combine(condition, img, gif):
    effect = gif[:,:,:3].copy().astype(np.float64)
    effect_edge = gif[:,:,:3].copy().astype(np.float64)
    imgOut = img.copy()
    edge1  = find_object_edges(condition,1)
    edge2  = find_object_edges(condition,2)
    effect[condition] = imgOut[condition]
    effect[edge1] = np.clip(effect_edge[edge1]*0.8 + imgOut[edge1]*0.2, 0, 255)
    effect[edge2] = np.clip(effect_edge[edge2]*0.9 + imgOut[edge2]*0.1, 0, 255)
    effect1 = effect.astype(np.uint8)
    return effect1

def high(img): 
    w = img.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j]:
                return i

segmentor = SelfiSegmentation()


cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(cv2.CAP_PROP_FPS, 60)

Threshold = 0.7 #tham số dùng để lấy đối đượng có khích với nền không

t = 0 # index của frame trong gif
count = 0 # đếm số frame ảnh của camera để thay đổi frame của gif

h = 0   #chiều cao hiện tại của đối tượng
hand = -1 #tham số dùng để check cho đổi background
time1 = time.time()
len_gif = len(bgr_gifList)
indexGif = 0


while True:
    success, img = cap.read()


    results = segmentor.selfieSegmentation.process(img)
    condition = np.stack(
    (results.segmentation_mask,), axis=-1) > Threshold
    condition1 = condition.reshape((480,640))


    time2 = time.time()
    if (time2 - time1) > 0.2:
        # Lấy chiều cao hiện tại
        now_h = high(condition1)
        time1 = time2
        # giá trị thay đổi của chiều cao
        change_h = abs(h - now_h)
        if change_h>80:
            h = now_h
            hand += 1
            # thay đổi background
            if hand == 2:
                if indexGif < (len_gif-1):
                    indexGif +=1
                    hand = 0
                    t = 0
                else:
                    indexGif = 0
                    hand = 0
                    t = 0

    if count == 3:
        t+=1
        count = 0
    count +=1
    if t == len(bgr_gifList[indexGif]):
        t = 0
    imgOut = combine(condition1, img, bgr_gifList[indexGif][t])
    cv2.imshow('Image', imgOut)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
