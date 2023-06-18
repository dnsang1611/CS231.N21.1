import cv2
import mediapipe as mp
import numpy as np
import copy
import matplotlib.pyplot as plt
from utils import detect_iris,calc_iris_point,calc_min_enc_losingCircle,\
draw_debug_image,draw_iris,check_eye_state,create_iris_mask,neon_effect
from face_mesh.face_mesh import FaceMesh
from iris_landmark.iris_landmark import IrisLandmark
from segmentation import get_segmentation_map
'''mediapipe -> RGB , opencv -> BGR'''
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# For static images:
IMAGE_FILES = ['images/test.png']
# Model load ################################################ ###############



for idx, file in enumerate(IMAGE_FILES):
  image = cv2.imread(file)
  # sizes of image
  w,h = image.shape[1],image.shape[0]
  # foreground_mask: BLACK is background/ WHITE is human
  foreground_mask = get_segmentation_map(image).astype(np.uint8)
  # create a copy
  frg = foreground_mask.copy()
  # black background
  black_img = np.zeros((h,w,3),dtype=np.uint8)
  # white background
  white_img = np.ones((h,w,3),dtype=np.uint8)
  # iris_mask : BLACK is iris/ WHITE is background
  iris_mask = create_iris_mask(image).astype('bool')
  # eyes: human eye (colored) extracted on black background
  eyes = np.where(iris_mask,black_img,image)
  # neon_eyes: eyes with glowing neon effect on black background
  neon_eyes = neon_effect(eyes)
  
  # neon_eyes_mask: neon_eyes segmentation mask on BLACK background, apply Otsu thresholding
  neon_eyes_mask_gray = cv2.cvtColor(neon_eyes, cv2.COLOR_BGR2GRAY)
  neon_eyes_mask_blurred = cv2.GaussianBlur(neon_eyes_mask_gray, (5, 5), 0)
  (T, neon_eyes_mask) = cv2.threshold(neon_eyes_mask_blurred, 0, 255, cv2.THRESH_OTSU)
  # use bitwise not (turns into WHITE background) to combine with foreground_mask
  neon_eyes_mask = cv2.bitwise_not(neon_eyes_mask)
  neon_eyes_mask = np.stack((neon_eyes_mask,neon_eyes_mask,neon_eyes_mask), axis=2)
  # final_mask
  final_mask = np.where(foreground_mask,neon_eyes_mask,frg).astype(np.uint8)
  # glowing eyes blended with seamlessclone
  # Taking a matrix of size 5 as the kernel
  kernel = np.ones((5, 5), np.uint8)
  mask = cv2.dilate(final_mask, kernel, iterations=1)
  glowing_eyes_blended = cv2.seamlessClone(image, neon_eyes, mask, (w//2+80,h//2), cv2.NORMAL_CLONE)
  glowing_eyes_blended = np.where(foreground_mask,glowing_eyes_blended,image)
  '''glowing_eyes_blended = np.where(final_mask,image,neon_eyes*0.5+image*0.5)'''
  

  # final result
  
  cv2.imwrite('neon_eyes.jpg',glowing_eyes_blended)
  
  

  
    
  

    

