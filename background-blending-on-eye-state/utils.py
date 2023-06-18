
import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp
import skimage.exposure

from face_mesh.face_mesh import FaceMesh
from iris_landmark.iris_landmark import IrisLandmark

def detect_iris(image, iris_detector, left_eye, right_eye):
    image_width, image_height = image.shape[1], image.shape[0]
    input_shape = iris_detector.get_input_shape()

    # left eye
    # Crop the image around the eyes
    left_eye_x1 = max(left_eye[0], 0)
    left_eye_y1 = max(left_eye[1], 0)
    left_eye_x2 = min(left_eye[2], image_width)
    left_eye_y2 = min(left_eye[3], image_height)
    left_eye_image = copy.deepcopy(image[left_eye_y1:left_eye_y2,
                                         left_eye_x1:left_eye_x2])
    # Iris detection
    eye_contour, iris = iris_detector(left_eye_image)
    # convert coordinates from relative to absolute
    left_iris = calc_iris_point(left_eye, eye_contour, iris, input_shape)

    # right eye
    # Crop the image around the eyes
    right_eye_x1 = max(right_eye[0], 0)
    right_eye_y1 = max(right_eye[1], 0)
    right_eye_x2 = min(right_eye[2], image_width)
    right_eye_y2 = min(right_eye[3], image_height)
    right_eye_image = copy.deepcopy(image[right_eye_y1:right_eye_y2,
                                          right_eye_x1:right_eye_x2])
    # Iris detection
    eye_contour, iris = iris_detector(right_eye_image)
    # convert coordinates from relative to absolute
    right_iris = calc_iris_point(right_eye, eye_contour, iris, input_shape)

    return left_iris, right_iris

# magical function, don't try to understand
def calc_iris_point(eye_bbox, eye_contour, iris, input_shape):
    iris_list = []
    for index in range(5):
        point_x = int(iris[index * 3] *
                      ((eye_bbox[2] - eye_bbox[0]) / input_shape[0]))
        point_y = int(iris[index * 3 + 1] *
                      ((eye_bbox[3] - eye_bbox[1]) / input_shape[1]))
        point_x += eye_bbox[0]
        point_y += eye_bbox[1]

        iris_list.append((point_x, point_y))

    return iris_list


def calc_min_enc_losingCircle(landmark_list):
    center, radius = cv.minEnclosingCircle(np.array(landmark_list))
    center = (int(center[0]), int(center[1]))
    radius = int(radius)

    return center, radius


def draw_debug_image(
    debug_image,
    left_iris,
    right_iris,
    left_center,
    left_radius,
    right_center,
    right_radius,
):
    # Rainbow: circumscribed yen
    cv.circle(debug_image, left_center, left_radius, (0, 255, 0), 2)
    cv.circle(debug_image, right_center, right_radius, (0, 255, 0), 2)

    '''# iris: landmark
    for point in left_iris:
        cv.circle(debug_image, (point[0], point[1]), 1, (0, 0, 255), 2)
    for point in right_iris:
        cv.circle(debug_image, (point[0], point[1]), 1, (0, 0, 255), 2)'''

    return debug_image

def draw_iris(
    image,
    left_center,
    right_center
):
    # FINAL RESULT
    cv.circle(image, left_center, 1, (0, 255, 0), 2)
    cv.circle(image, right_center, 1, (0, 255, 0), 2)

    return image

def create_iris_mask(image):
    face_mesh = FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5
    )
    iris_detector = IrisLandmark()
    # Create mask
    mask = np.full((image.shape[0],image.shape[1],3),255)
    # Face Mesh detection, face_result -> <List(landmark_x, landmark_y, landmark.z, landmark.visibility, landmark.presence)>
    face_results = face_mesh(image)

    for face_result in face_results:
        # Calculate bounding box around eyes
        left_eye, right_eye = face_mesh.calc_around_eye_bbox(face_result)
        # Iris detection
        left_iris, right_iris = detect_iris(image, iris_detector, left_eye, right_eye)
        # Calculate the circumcircle of the iris, USE THESE COORDINATES
        left_center, left_radius = calc_min_enc_losingCircle(left_iris)
        right_center, right_radius = calc_min_enc_losingCircle(right_iris)
        
        # Draw a circle of black color of thickness -1 px
        mask = cv.circle(mask, left_center, left_radius+15, (0,0,0), -1)
        mask = cv.circle(mask, right_center, right_radius+15, (0,0,0), -1)

    return mask.astype(np.uint8)


def neon_effect(img):
    # convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # threshold
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    # do morphology gradient to get edges and invert so black edges on white background
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    edges = cv.morphologyEx(thresh, cv.MORPH_GRADIENT, kernel)
    edges = 255 - edges

    # get distance transform
    dist = edges.copy()
    distance = cv.distanceTransform(dist, distanceType=cv.DIST_L2, maskSize=3)

    # stretch to full dynamic range and convert to uint8 as 3 channels
    stretch = skimage.exposure.rescale_intensity(distance, in_range=('image'), out_range=(0,255))

    # invert 
    stretch = 255 - stretch
    max_stretch = np.amax(stretch)

    # normalize to range 0 to 1 by dividing by max_stretch
    stretch = (stretch/max_stretch)

    # attenuate with power law 
    pow = 32
    attenuate = np.power(stretch, pow)
    attenuate = cv.merge([attenuate,attenuate,attenuate])

    # create a BLUE image the size of the input
    color_img = np.full_like(img, (255,0,0), dtype=np.float32)

    # multiply the color image with the attenuated distance image
    glow = (color_img * attenuate).clip(0,255).astype(np.uint8)

    return glow


'''Return True if eyes in the image are open, else return False'''
def check_eye_state(image):
    w,h = image.shape[:2]
    LEFT_EYE = [362,263,385,380,386,374] #  index 1,2 -> ngang , 3,4 -> doc
    RIGHT_EYE = [33,133,159,145,158,153]
    # Load face mesh model
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    )

    # Process the image and obtain face mesh landmarks
    results = mp_face_mesh.process(image)

    # Check if any face landmarks are detected
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract eye landmarks for the left eye
        left_eye_points = np.array([[landmark.x * w, landmark.y * h] for landmark in face_landmarks.landmark])[LEFT_EYE]
        
        # Extract eye landmarks for the right eye
        right_eye_points = np.array([[landmark.x * w, landmark.y * h] for landmark in face_landmarks.landmark])[RIGHT_EYE]
        
        # Calculate eye openness ratio for the left eye
        left_eye_openness = (np.abs(left_eye_points[3][1]-left_eye_points[2][1])+np.abs(left_eye_points[5][1]-left_eye_points[4][1]))/(2*np.abs(left_eye_points[1][0]-left_eye_points[0][0]))
        
        # Calculate eye openness ratio for the right eye
        right_eye_openness = (np.abs(right_eye_points[3][1]-right_eye_points[2][1])+np.abs(right_eye_points[5][1]-right_eye_points[4][1]))/(2*np.abs(right_eye_points[1][0]-left_eye_points[0][0]))
        
        # Define a threshold to determine if the eyeSs are open or closed
        threshold = 1.0
        # print("Left eye: {}".format(left_eye_openness))
        # Return True if both eyes are open, else return False
        if left_eye_openness < threshold and right_eye_openness < threshold:
            return False
        else:
            return True
    else:
        return False

'''Plot the eye state to get right threshold'''
def plot_eye_state(image, array):
    w, h = image.shape[:2]
    LEFT_EYE = [362, 263, 385, 380, 386, 374]  # index 1,2 -> ngang , 3,4 -> doc
    RIGHT_EYE = [33, 133, 159, 145, 158, 153]
    # Load face mesh model
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    )

    # Process the image and obtain face mesh landmarks
    results = mp_face_mesh.process(image)

    # Check if any face landmarks are detected
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Extract eye landmarks for the left eye
        left_eye_points = np.array([[landmark.x * w, landmark.y * h] for landmark in face_landmarks.landmark])[LEFT_EYE]

        # Extract eye landmarks for the right eye
        right_eye_points = np.array([[landmark.x * w, landmark.y * h] for landmark in face_landmarks.landmark])[RIGHT_EYE]

        # Calculate eye openness ratio for the left eye
        left_eye_openness = (np.abs(left_eye_points[3][1] - left_eye_points[2][1]) + np.abs(
            left_eye_points[5][1] - left_eye_points[4][1])) / (
                                     2 * np.abs(left_eye_points[1][0] - left_eye_points[0][0]))

        # Calculate eye openness ratio for the right eye
        right_eye_openness = (np.abs(right_eye_points[3][1] - right_eye_points[2][1]) + np.abs(
            right_eye_points[5][1] - right_eye_points[4][1])) / (
                                      2 * np.abs(right_eye_points[1][0] - left_eye_points[0][0]))

        # Append into the array (left eye)
        array = np.append(array, [[left_eye_openness]])
    
    return array

    
        
    
                     
