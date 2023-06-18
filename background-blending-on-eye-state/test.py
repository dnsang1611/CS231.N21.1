import cv2
import mediapipe as mp
import numpy as np
import copy
import matplotlib.pyplot as plt
from utils import check_eye_state
from segmentation import get_segmentation_map
from face_mesh.face_mesh import FaceMesh
import imageio
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
            # creating the alpha mask
            alpha = np.zeros_like(foreground)
            gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
            alpha[:, :, 0] = gray
            alpha[:, :, 1] = gray
            alpha[:, :, 2] = gray

            # converting uint8 to float type
            foreground = foreground.astype(float)
            

            background = background.astype(float)

            # normalizing the alpha mask inorder
            # to keep intensity between 0 and 1
            alpha = alpha.astype(float) / 255

            # multiplying the foreground
            # with alpha matte
            foreground = cv2.multiply(alpha, foreground)

            # multiplying the background
            # with (1 - alpha)
            background = cv2.multiply(1.0 - alpha, background)

            # adding the masked foreground
            # and background together
            outImage = cv2.add(foreground, background)
            # ... (code)
            # Apply segmentation
            mask = get_segmentation_map(init_foreground).astype(np.uint8)
            # preprocess the maks so that cv2.seamlessClone can process it
            # Taking a matrix of size 5 as the kernel
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask*255, kernel, iterations=1)
            
            merged = np.where(mask, init_foreground, outImage).astype(np.uint8)
            '''merged = cv2.seamlessClone(init_foreground,merged,mask,(w//2,h//2), cv2.NORMAL_CLONE)'''
            # showing the masked output video
            result.append(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB))
            

        else:
            result.append(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
            
    # if the actual video is over then there's
    # nothing in the foreground array thus
    # breaking from the while loop
    else:
        break


fg.release()
cv2.destroyAllWindows()
imageio.mimsave('result.gif', result, 'GIF', duration=0.04)


print('Video Blending is done perfectly')
