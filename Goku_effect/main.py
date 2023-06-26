import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageSequence

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Read gif file
effect_gif = Image.open('goku.gif')
space_gif = Image.open('space.gif')

# Save each frame of gif to img
index = 1
for frame in ImageSequence.Iterator(effect_gif):
    frame.save("effect%d.png" % index)
    index += 1

# Load effect imgs
effect_img = []
for i in range(1, index, 1):
    temp = cv2.imread("effect%d.png" % i)
    temp = cv2.resize(temp, (339, 480))
    effect_img.append(temp)

# Save each frame of gif to img
index = 1
for frame in ImageSequence.Iterator(space_gif):
    frame.save("space%d.png" % index)
    index += 1

# Load effect imgs
space_img = []
for i in range(1, index, 1):
    temp = cv2.imread("space%d.png" % i)
    temp = cv2.resize(temp, (640, 480))
    space_img.append(temp)

# Load hair
hair_img = cv2.imread("hair.png")

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return round(angle, 2)

cap = cv2.VideoCapture(0)

# Set current time
time = 0

## Setup mediapipe instance
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True) as pose:
    while cap.isOpened():
        time += 1

        ret, frame = cap.read()
        print(frame.shape)
        width, height, _ = frame.shape

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]
            right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]

            # Calculate some coordinates
            center = [(left_hip[0]+right_hip[0])/2, (left_hip[1]+right_hip[1])/2]
            center = np.multiply(center, [height, width]).astype('int')
            head_center = [(left_eye[0]+right_eye[0])/2, (left_eye[1]+right_eye[1])/2]
            head_center = np.multiply(head_center, [height, width]).astype('int')
            nose = np.multiply(nose, [height, width]).astype('int')

            left_ear = np.multiply(left_ear, [height, width]).astype('int')
            right_ear = np.multiply(right_ear, [height, width]).astype('int')
            disc = int(np.sqrt((left_ear[0]-right_ear[0])**2 + (left_ear[1]-right_ear[1])**2))
            disc = disc * 2
            hair_img = cv2.resize(hair_img, (disc, disc))

            # Calculate angle
            angle1 = calculate_angle(right_hip, right_shoulder, right_elbow)
            angle2 = calculate_angle(left_hip, left_shoulder, left_elbow)

            # Visualize angle
            # cv2.putText(image, str(angle1),
            #             tuple(np.multiply(right_shoulder, [height, width]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #             )
            # cv2.putText(image, str(angle2),
            #             tuple(np.multiply(left_shoulder, [height, width]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #             )

            # Visualize effect
            if angle2 > 25 and angle2 < 50 and angle1 > 25 and angle1 < 50:
                # # Set bg
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5
                bg_image = space_img[time%30]
                image = np.where(condition, image, bg_image)

                # Set effects and hair
                if time % 4 == 0:
                    image[0:480, center[0]+30-180:center[0]+30+159, :] = cv2.add(image[0:480, center[0]+30-180:center[0]+30+159, :], effect_img[0])
                elif time % 4 == 1:
                    image[0:480, center[0]-180:center[0]+159, :] = cv2.add(image[0:480, center[0]-180:center[0]+159, :], effect_img[1])
                elif time % 4 == 2:
                    image[0:480, center[0]+30-180:center[0]+30+159, :] = cv2.add(image[0:480, center[0]+30-180:center[0]+30+159, :], effect_img[2])
                elif time % 4 == 3:
                    image[0:480, center[0]-180:center[0]+159, :] = cv2.add(image[0:480, center[0]-180:center[0]+159, :], effect_img[3])
                else:
                    pass

                if head_center[1] - disc >= 0 and head_center[0] - disc // 2 >= 0:
                    image[head_center[1] - disc:head_center[1],
                    head_center[0] - disc // 2:head_center[0] - disc // 2 + disc, :] = cv2.add(
                        image[head_center[1] - disc:head_center[1],
                        head_center[0] - disc // 2:head_center[0] - disc // 2 + disc, :], hair_img)
                elif head_center[1] - disc < 0:
                    new_x = abs(head_center[1] - disc)
                    image[0:head_center[1], head_center[0] - disc // 2:head_center[0] - disc // 2 + disc,
                    :] = cv2.add(
                        image[0:head_center[1], head_center[0] - disc // 2:head_center[0] - disc // 2 + disc, :],
                        hair_img[new_x:])
                else:
                    pass
            else:
                pass

        except:
            pass

        # Render detections
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        #                           mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        #                           )

        cv2.imshow('Super Xaiyan Effect', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()