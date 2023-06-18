import cv2
import mediapipe as mp
import numpy as np
import copy
import matplotlib.pyplot as plt
from utils import detect_iris,calc_iris_point,calc_min_enc_losingCircle,draw_debug_image,draw_iris,check_eye_state,\
neon_effect,create_iris_mask
from segmentation import get_segmentation_map
from face_mesh.face_mesh import FaceMesh
from iris_landmark.iris_landmark import IrisLandmark
import imageio

iris_detector = IrisLandmark()
# face mesh
face_mesh = FaceMesh(
  max_num_faces=1,
  min_detection_confidence=0.5
)
# mediapipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# assigning path of foreground video
path_1 = r"videos/man.mp4"
fg = cv2.VideoCapture(path_1)

# assigning path of background video
path_2 = r"videos/4K_3.mp4"
bg = cv2.VideoCapture(path_2)
w,h = 1741,874
result = []

while (True):
    # print("Frame no.{}".format(count))
    # Reading the two input videos
    # we have taken "ret" here because the duration
    # of bg video is greater than fg video
    ret, foreground = fg.read()
    if not ret:
        break
    # foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB )
    # foreground = foreground.astype(np.uint8)
    foreground = cv2.resize(foreground, (w,h))
    init_foreground = foreground.copy()
    # if in your case the situation is opposite
    # then take the "ret" for bg video
    _, background = bg.read()
    background = cv2.resize(background, (w,h))
    init_background = background.copy()
    # if foreground array is not empty which
    # means actual video is still going on
    if ret:

        '''If the person opens his/her eyes, find iris coordinates,
        get segmentation mask and apply background effect'''
        # Face Mesh detection, face_result<List>
        face_results = face_mesh(foreground)
        # If the person opens his/her eyes
        if check_eye_state(foreground) == True:
            # foreground_mask: BLACK is background/ WHITE is human
            foreground_mask = get_segmentation_map(foreground).astype(np.uint8)
            # create a copy
            frg = foreground_mask.copy()
            # black background
            black_img = np.zeros((h,w,3),dtype=np.uint8)
            # white background
            white_img = np.ones((h,w,3),dtype=np.uint8)
            # iris_mask : BLACK is iris/ WHITE is background
            iris_mask = create_iris_mask(foreground).astype('bool')
            
            # eyes: human eye (colored) extracted on black background
            eyes = np.where(iris_mask,black_img,foreground)
            
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
            # Apply segmentation
            glowing_eyes_blended = cv2.seamlessClone(foreground, neon_eyes, mask, (w//2+80,h//2), cv2.NORMAL_CLONE)
            merged = np.where(foreground_mask,glowing_eyes_blended,init_background)
            
            '''# preprocess the maks so that cv2.seamlessClone can process it
            # Taking a matrix of size 5 as the kernel
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask*255, kernel, iterations=1)
            
            
            merged = cv2.seamlessClone(glowing_eyes_blended,init_background,mask,(w//2,h//2), cv2.NORMAL_CLONE)'''
            
            
            # showing the masked output video
            result.append(cv2.cvtColor(merged.astype(np.uint8), cv2.COLOR_BGR2RGB))

        else:
            result.append(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
            
    # if the actual video is over then there's
    # nothing in the foreground array thus
    # breaking from the while loop
    else:
        break


fg.release()
cv2.destroyAllWindows()
imageio.mimsave('result.gif', result, 'GIF')


print('Video Blending is done perfectly')
