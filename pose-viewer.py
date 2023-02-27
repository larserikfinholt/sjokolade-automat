import cv2
import csv
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
from mediapipe.framework.formats import landmark_pb2

posesNames = ["RaiseLeftArm", "RaiseRightArm", "RiseBothArms"]
poseNo = 0
poses=[]


for poseName in posesNames:
 
    landmarks = []
    with open("./data/" + poseName+".csv", "r") as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            values = []
            for i in range(0, len(row), 4):
                values.append(row[i:i + 4])

            landmark_list = landmark_pb2.NormalizedLandmarkList()
            for v in values:
                landmark = landmark_pb2.NormalizedLandmark()
                landmark.x= float(v[0])
                landmark.y =float( v[1])
                landmark.z =float( v[2])
                landmark.visibility =float( v[3])
                landmark_list.landmark.append(landmark)
            
            # print(landmark_dict)
            landmarks.append(landmark_list)

  

    # Loop over some frames
    for i in range(len(landmarks)):
        # Generate a blank image
        frame = np.zeros((480, 640, 3), np.uint8)

        # Draw something on the image
        cv2.putText(frame, str(i), (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)


        # Draw the pose annotation on the image.
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # print(landmarks[i])
        try:
            mp_drawing.draw_landmarks(
                    frame,
                    landmarks[i],
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        except Exception as e:
            print(e)

        cv2.imshow('MediaPipe Pose', frame)

        cv2.waitKey(10)