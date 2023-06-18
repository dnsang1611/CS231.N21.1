import cv2
import mediapipe as mp
import numpy as np

def get_segmentation_map(image, bg_color=(0,0,0), model_complexity=2):
    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=model_complexity,
        enable_segmentation=True,
        refine_face_landmarks=True) as holistic:
      
        image_height, image_width, _ = image.shape
        
        # Convert the BGR image to RGB before processing.
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        annotated_image = image.copy()

        # Draw segmentation on the image.
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5
        '''bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = bg_color
        annotated_image = np.where(condition, annotated_image, bg_image)

        annotated_image_file = 'seg_map.png'
        cv2.imwrite(annotated_image_file, annotated_image)'''

        return condition
        
# IMAGE_PATH = 'images/test.png'
# alpha = 0.2
# bg = cv2.imread('images/thunder.jpg')
# fg = cv2.imread(IMAGE_PATH)
# print(fg.shape)
# bg = np.resize(bg,(fg.shape[0],fg.shape[1],fg.shape[2]))
# mask = get_segmentation_map(IMAGE_PATH)
# merged = np.where(mask, fg, fg*alpha+bg*(1-alpha))
# SAVE_PATH = 'seg_map.jpg'
# cv2.imwrite(SAVE_PATH, merged)