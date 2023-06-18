import cv2
import mediapipe as mp
import numpy as np
import copy
import matplotlib.pyplot as plt
from utils import detect_iris,calc_iris_point,calc_min_enc_losingCircle,draw_debug_image,draw_iris,check_eye_state,plot_eye_state
from segmentation import get_segmentation_map
from face_mesh.face_mesh import FaceMesh
from iris_landmark.iris_landmark import IrisLandmark
import imageio
# face mesh
face_mesh = FaceMesh(
  max_num_faces=1,
  min_detection_confidence=0.5
)
iris_detector = IrisLandmark()
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
array = np.array([])
LEFT_EYE = [362,263,385,380,386,374] #  index 1,2 -> ngang , 3,4 -> doc
RIGHT_EYE = [33,133,159,145,158,153]
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
        array = plot_eye_state(foreground,array)

            
    # if the actual video is over then there's
    # nothing in the foreground array thus
    # breaking from the while loop
    else:
        break


fg.release()
cv2.destroyAllWindows()
M = np.max(array)
m = np.min(array)

plt.plot(array, color = 'b')
plt.axhline((M+m)/2, color='r')
plt.xlabel("Frame no.")
plt.ylabel("EAR")
plt.title("Brute force plot")
plt.legend()
# Threshold should choose is (M+m)/2 = 1.030327901822079 => choose 1.0
print((M+m)/2)
plt.show()


print('Video Blending is done perfectly')
